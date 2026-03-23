#!/usr/bin/env bash
#
# karen-preflight.sh — Mechanical checks gate for the Karen Stop hook.
#
# Runs cargo fmt --check, cargo clippy, and cargo test.
# If ANY check fails, blocks Claude from stopping (exit 2) and reports
# what failed so the agent can fix it before declaring done.
#
# Reads JSON on stdin (Claude Code Stop hook protocol).
# Outputs JSON with decision + reason on failure, or exits 0 on success.
#
set -euo pipefail

INPUT="$(cat)"

# --- Infinite-loop guard ---------------------------------------------------
# If this Stop hook already fired once and Claude continued, let it stop
# on the second pass to avoid an infinite review loop.
STOP_HOOK_ACTIVE="$(echo "$INPUT" | jq -r '.stop_hook_active // false')"
if [ "$STOP_HOOK_ACTIVE" = "true" ]; then
  exit 0
fi

# --- Resolve working directory ----------------------------------------------
CWD="$(echo "$INPUT" | jq -r '.cwd // empty')"
if [ -z "$CWD" ]; then
  CWD="${CLAUDE_PROJECT_DIR:-.}"
fi
cd "$CWD"

# Only run in a Rust project (Cargo.toml present)
if [ ! -f Cargo.toml ]; then
  exit 0
fi

FAILURES=""

# --- cargo fmt --check ------------------------------------------------------
if ! FMT_OUTPUT="$(cargo fmt --check 2>&1)"; then
  FAILURES="${FAILURES}FORMATTING: cargo fmt --check failed.\n${FMT_OUTPUT}\n\n"
fi

# --- cargo clippy -----------------------------------------------------------
if ! CLIPPY_OUTPUT="$(cargo clippy -- -D warnings 2>&1)"; then
  FAILURES="${FAILURES}LINTING: cargo clippy -- -D warnings failed.\n${CLIPPY_OUTPUT}\n\n"
fi

# --- cargo test -------------------------------------------------------------
if ! TEST_OUTPUT="$(cargo test 2>&1)"; then
  FAILURES="${FAILURES}TESTS: cargo test failed.\n${TEST_OUTPUT}\n\n"
fi

# --- Verdict ----------------------------------------------------------------
if [ -n "$FAILURES" ]; then
  # Truncate to avoid blowing up the hook response (keep last 3000 chars)
  TRIMMED="$(echo -e "$FAILURES" | tail -c 3000)"
  jq -n --arg reason "$TRIMMED" '{
    "decision": "block",
    "reason": ("Mechanical checks failed. Fix these before stopping:\n\n" + $reason)
  }'
  exit 0
fi

# All checks passed — let the prompt hook (Karen) do the subjective review.
exit 0
