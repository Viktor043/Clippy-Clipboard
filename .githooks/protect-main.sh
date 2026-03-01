#!/bin/bash
# Pre-push hook: prevent direct pushes to main
branch=$(git symbolic-ref --short HEAD 2>/dev/null)
if [ "$branch" = "main" ]; then
    echo "ERROR: Direct push to 'main' is blocked."
    echo "Create a feature branch and push that instead:"
    echo ""
    echo "  git checkout -b feat/my-feature"
    echo "  git push origin feat/my-feature"
    echo ""
    exit 1
fi
exit 0
