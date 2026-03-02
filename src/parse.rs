use std::path::Path;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor};

pub struct RawChunk {
    pub name: String,
    pub source: String,
    pub embed_text: String,
    pub kind: String,
}

struct LangConfig {
    language: Language,
    chunk_query: &'static str,
}

fn lang_for_path(path: &Path) -> Option<LangConfig> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");

    // Starlark BUILD files have no meaningful extension — match by filename.
    let effective_ext = match filename {
        "BUILD" | "BUILD.bazel" | "WORKSPACE" | "WORKSPACE.bazel" => "bzl",
        _ => ext,
    };

    match effective_ext {
        "rs" => Some(LangConfig {
            language: tree_sitter_rust::LANGUAGE.into(),
            chunk_query: r#"
                (function_item name: (identifier) @name) @chunk
                (struct_item name: (type_identifier) @name) @chunk
                (enum_item name: (type_identifier) @name) @chunk
                (trait_item name: (type_identifier) @name) @chunk
                (impl_item) @chunk
            "#,
        }),
        "py" => Some(LangConfig {
            language: tree_sitter_python::LANGUAGE.into(),
            chunk_query: r#"
                (function_definition name: (identifier) @name) @chunk
                (class_definition name: (identifier) @name) @chunk
            "#,
        }),
        "go" => Some(LangConfig {
            language: tree_sitter_go::LANGUAGE.into(),
            chunk_query: r#"
                (function_declaration name: (identifier) @name) @chunk
                (method_declaration name: (field_identifier) @name) @chunk
                (type_declaration (type_spec name: (type_identifier) @name)) @chunk
            "#,
        }),
        "yaml" | "yml" => Some(LangConfig {
            language: tree_sitter_yaml::LANGUAGE.into(),
            chunk_query: r#"
                (block_mapping_pair key: (flow_node) @name) @chunk
            "#,
        }),
        "bzl" => Some(LangConfig {
            language: tree_sitter_starlark::LANGUAGE.into(),
            chunk_query: r#"
                (function_definition name: (identifier) @name) @chunk
                (expression_statement
                    (call function: (identifier) @name)) @chunk
            "#,
        }),
        _ => None,
    }
}

pub fn parse_file(path: &Path, source: &str) -> Vec<RawChunk> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    if matches!(ext, "md" | "markdown") {
        return parse_markdown(source);
    }

    match lang_for_path(path) {
        Some(config) => parse_with_treesitter(path, source, config),
        None => vec![],
    }
}

fn parse_with_treesitter(path: &Path, source: &str, config: LangConfig) -> Vec<RawChunk> {
    let mut parser = Parser::new();
    parser
        .set_language(&config.language)
        .expect("tree-sitter language load must not fail");

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => {
            eprintln!("tree-sitter: parse failed for {}", path.display());
            return vec![];
        }
    };

    let query = match Query::new(&config.language, config.chunk_query) {
        Ok(q) => q,
        Err(e) => {
            eprintln!("tree-sitter: invalid query: {e:?}");
            return vec![];
        }
    };

    let chunk_idx = match query.capture_index_for_name("chunk") {
        Some(i) => i,
        None => {
            eprintln!("tree-sitter: query missing @chunk capture");
            return vec![];
        }
    };
    let name_idx = query.capture_index_for_name("name");

    let mut cursor = QueryCursor::new();
    let mut chunks = Vec::new();

    let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
    while let Some(m) = matches.next() {
        let chunk_cap = match m.captures.iter().find(|c| c.index == chunk_idx) {
            Some(c) => c,
            None => continue,
        };
        let name_cap = name_idx.and_then(|ni| m.captures.iter().find(|c| c.index == ni));

        let node = chunk_cap.node;
        let source_text = &source[node.byte_range()];
        let embed_text = if source_text.len() > 2000 {
            let end = (0..=2000)
                .rev()
                .find(|&i| source_text.is_char_boundary(i))
                .unwrap_or(0);
            &source_text[..end]
        } else {
            source_text
        };
        let name = name_cap
            .map(|n| source[n.node.byte_range()].to_string())
            .unwrap_or_else(|| "(anonymous)".to_string());

        chunks.push(RawChunk {
            name,
            source: source_text.to_string(),
            embed_text: embed_text.to_string(),
            kind: node.kind().to_string(),
        });
    }

    chunks
}

/// Splits a Markdown document into per-heading sections.
/// Each ATX heading (# / ## / ### …) starts a new chunk; text before the first
/// heading is emitted as a "(preamble)" chunk if non-empty.
fn parse_markdown(source: &str) -> Vec<RawChunk> {
    let mut chunks = Vec::new();
    let mut section_name = String::new();
    let mut section_lines: Vec<&str> = Vec::new();

    for line in source.lines() {
        if line.starts_with('#') {
            let heading_text = line.trim_start_matches('#').trim();
            if !heading_text.is_empty() {
                flush_markdown_section(&section_name, &section_lines, &mut chunks);
                section_name = heading_text.to_string();
                section_lines = vec![line];
                continue;
            }
        }
        section_lines.push(line);
    }
    flush_markdown_section(&section_name, &section_lines, &mut chunks);

    chunks
}

fn flush_markdown_section(name: &str, lines: &[&str], chunks: &mut Vec<RawChunk>) {
    let joined = lines.join("\n");
    let trimmed = joined.trim().to_string();
    if trimmed.is_empty() {
        return;
    }
    chunks.push(RawChunk {
        name: if name.is_empty() { "(preamble)".to_string() } else { name.to_string() },
        source: joined,
        embed_text: trimmed,
        kind: "section".to_string(),
    });
}
