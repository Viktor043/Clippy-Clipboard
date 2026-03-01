#!/bin/bash
# Post-commit hook: log commit to agent audit JSONL
MSG=$(git log --format=%s -1 2>/dev/null || echo "unknown")
~/.openclaw/workspace/agent-audit.sh git commit success "$MSG"
