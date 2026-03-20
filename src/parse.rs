use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Node, Parser, Query, QueryCursor};

/// Semantic kind of a parsed chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
pub enum ChunkKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Class,
    Method,
    TypeDecl,
    BlockMapping,
    Expression,
    Section,
    Module,
}

impl ChunkKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Impl => "impl",
            Self::Class => "class",
            Self::Method => "method",
            Self::TypeDecl => "type_decl",
            Self::BlockMapping => "block_mapping",
            Self::Expression => "expression",
            Self::Section => "section",
            Self::Module => "module",
        }
    }

    pub fn from_db_str(s: &str) -> Self {
        match s {
            "function" => Self::Function,
            "struct" => Self::Struct,
            "enum" => Self::Enum,
            "trait" => Self::Trait,
            "impl" => Self::Impl,
            "class" => Self::Class,
            "method" => Self::Method,
            "type_decl" => Self::TypeDecl,
            "block_mapping" => Self::BlockMapping,
            "expression" => Self::Expression,
            "section" => Self::Section,
            "module" => Self::Module,
            _ => Self::Section,
        }
    }

    fn from_ts_node(node_kind: &str) -> Self {
        match node_kind {
            "function_item" | "function_definition" | "function_declaration" => Self::Function,
            "struct_item" => Self::Struct,
            "enum_item" => Self::Enum,
            "trait_item" => Self::Trait,
            "impl_item" => Self::Impl,
            "class_definition" => Self::Class,
            "method_declaration" => Self::Method,
            "type_declaration" | "type_spec" => Self::TypeDecl,
            "block_mapping_pair" => Self::BlockMapping,
            "expression_statement" => Self::Expression,
            "module" | "source_file" | "mod_item" => Self::Module,
            _ => Self::Section,
        }
    }
}

impl fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Function => write!(f, "function"),
            Self::Struct => write!(f, "struct"),
            Self::Enum => write!(f, "enum"),
            Self::Trait => write!(f, "trait"),
            Self::Impl => write!(f, "impl"),
            Self::Class => write!(f, "class"),
            Self::Method => write!(f, "method"),
            Self::TypeDecl => write!(f, "type"),
            Self::BlockMapping => write!(f, "mapping"),
            Self::Expression => write!(f, "expression"),
            Self::Section => write!(f, "section"),
            Self::Module => write!(f, "module"),
        }
    }
}

#[derive(Clone)]
pub struct RawChunk {
    pub name: String,
    pub source: String,
    pub kind: ChunkKind,
}

struct LangConfig {
    language: Language,
    chunk_query: &'static str,
}

fn lang_for_path(path: &Path) -> Option<LangConfig> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("");

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
                (impl_item type: (_) @name) @chunk
                (mod_item name: (identifier) @name) @chunk
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
        "c" | "h" => Some(LangConfig {
            language: tree_sitter_c::LANGUAGE.into(),
            chunk_query: r#"
                (function_definition declarator: (function_declarator declarator: (identifier) @name)) @chunk
                (struct_specifier name: (type_identifier) @name) @chunk
                (enum_specifier name: (type_identifier) @name) @chunk
                (type_definition declarator: (type_identifier) @name) @chunk
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
        None => {
            log::warn!("[parse] no parser for {:?} — file will be skipped (remove extension from config or add a parser)", path);
            vec![]
        }
    }
}

const CHUNK_THRESHOLD: usize = 2000;

fn parse_with_treesitter(path: &Path, source: &str, config: LangConfig) -> Vec<RawChunk> {
    let mut parser = Parser::new();
    parser.set_language(&config.language).expect("TS lang load failed");

    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return vec![],
    };

    let mut landmark_names = HashMap::new();
    if let Ok(query) = Query::new(&config.language, config.chunk_query) {
        let mut cursor = QueryCursor::new();
        let name_idx = query.capture_index_for_name("name");
        let chunk_idx = query.capture_index_for_name("chunk");

        if let (Some(c_idx), Some(n_idx)) = (chunk_idx, name_idx) {
            let mut matches = cursor.matches(&query, tree.root_node(), source.as_bytes());
            while let Some(m) = matches.next() {
                let chunk_node = m.captures.iter().find(|c| c.index == c_idx).map(|c| c.node);
                let name_node = m.captures.iter().find(|c| c.index == n_idx).map(|c| c.node);
                if let (Some(cn), Some(nn)) = (chunk_node, name_node) {
                    let id = cn.id();
                    let name = source[nn.byte_range()].to_string();
                    landmark_names.insert(id, name);
                }
            }
        }
    }

    let chunker = Chunker {
        source,
        landmark_names,
        threshold: CHUNK_THRESHOLD,
    };

    let filename = path.file_name().and_then(|f| f.to_str()).unwrap_or("file");
    let mut results = Vec::new();
    chunker.chunk_iterative(tree.root_node(), filename, &mut results);
    results
}

struct Chunker<'a> {
    source: &'a str,
    landmark_names: HashMap<usize, String>,
    threshold: usize,
}

struct Task<'a> {
    node: Node<'a>,
    parent_name: String,
}

impl<'a> Chunker<'a> {
    fn chunk_iterative(&self, root_node: Node, initial_name: &str, results: &mut Vec<RawChunk>) {
        let mut stack = vec![Task {
            node: root_node,
            parent_name: initial_name.to_string(),
        }];

        while let Some(task) = stack.pop() {
            let node = task.node;
            let range = node.byte_range();
            let kind = ChunkKind::from_ts_node(node.kind());

            let landmark_name = self.landmark_names.get(&node.id());
            let is_container = matches!(kind, ChunkKind::Impl | ChunkKind::Class | ChunkKind::Module | ChunkKind::Trait | ChunkKind::Section);

            let current_name = if let Some(name) = landmark_name {
                if task.parent_name == *name || task.parent_name.ends_with(name) {
                    task.parent_name.clone()
                } else {
                    format!("{} > {}", task.parent_name, name)
                }
            } else {
                task.parent_name.clone()
            };

            // 1. Terminal condition: Small enough node.
            if range.len() <= self.threshold {
                self.emit_node(node, &current_name, results);
                continue;
            }

            // 2. Terminal condition: Terminal landmark (e.g. giant Function).
            if landmark_name.is_some() && !is_container {
                self.emit_node(node, &current_name, results);
                continue;
            }

            // 3. Recursive condition: Too big and (not a landmark OR a container landmark).
            let mut cursor = node.walk();
            let children: Vec<Node> = node.children(&mut cursor).collect();

            if children.is_empty() {
                self.emit_node(node, &current_name, results);
                continue;
            }

            // Partition children: group small non-landmarks into buckets, push others to stack.
            let mut i = 0;
            let mut sub_tasks = Vec::new();
            while i < children.len() {
                let child = children[i];
                let child_len = child.byte_range().len();

                if self.landmark_names.contains_key(&child.id()) || child_len > self.threshold {
                    // Landmark or massive child: process individually.
                    sub_tasks.push(Task {
                        node: child,
                        parent_name: current_name.clone(),
                    });
                    i += 1;
                } else {
                    // Bucket adjacent small non-landmark children.
                    let start_byte = child.start_byte();
                    let mut end_byte = child.end_byte();
                    let mut j = i + 1;
                    while j < children.len() {
                        let next_child = children[j];
                        if self.landmark_names.contains_key(&next_child.id()) || (next_child.end_byte() - start_byte) > self.threshold {
                            break;
                        }
                        end_byte = next_child.end_byte();
                        j += 1;
                    }
                    
                    let bucket_source = &self.source[start_byte..end_byte];
                    results.push(RawChunk {
                        name: current_name.clone(),
                        source: bucket_source.to_string(),
                        kind: ChunkKind::from_ts_node(node.kind()),
                    });
                    i = j;
                }
            }
            
            // Push sub-tasks in reverse order to maintain left-to-right processing.
            for t in sub_tasks.into_iter().rev() {
                stack.push(t);
            }
        }
    }

    fn emit_node(&self, node: Node, name: &str, results: &mut Vec<RawChunk>) {
        let range = node.byte_range();
        let source_text = &self.source[range];
        results.push(RawChunk {
            name: name.to_string(),
            source: source_text.to_string(),
            kind: ChunkKind::from_ts_node(node.kind()),
        });
    }
}

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
        source: trimmed,
        kind: ChunkKind::Section,
    });
}
