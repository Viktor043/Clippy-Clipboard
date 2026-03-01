# CLAUDE.md — Agent Policy for Clippy

## Project Overview
Clippy is a macOS clipboard manager built with Swift 5 and SpriteKit, targeting macOS 14.0+.
The Xcode project is generated from `project.yml` using XcodeGen. Bundle ID: `com.snrgystudios.clippy`.

## Repository
- Remote: Viktor043/Clippy-Project (private)
- Default branch: main
- All commits MUST be signed (SSH signing configured globally)

## Code Standards
- Language: Swift 5.0
- Minimum deployment target: macOS 14.0
- Project structure: XcodeGen (`project.yml`) — never edit `.xcodeproj` directly
- Commit style: Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`)
- Pre-commit hooks are mandatory — never bypass with `--no-verify`

## Directory Layout
```
Clippy/App/              — App entry point, AppDelegate, Info.plist
Clippy/ClippyCharacter/  — SpriteKit character controller
Clippy/Models/           — Data models
Clippy/Services/         — Business logic and services
Clippy/Utilities/        — Constants, helpers
Clippy/Views/            — SwiftUI/AppKit views
Assets.xcassets/         — Asset catalog
Resources/               — Bundled resources (sprite agents)
Scripts/                 — Build and conversion scripts
Chip/                    — ML subproject
BatteryBar/              — Battery widget subproject
BrightnessControl/       — Brightness control subproject
clippy-web/              — Web module subproject
```

## Autonomous Actions (ALLOWED without approval)
- Read any file in the repository
- Search code, grep, glob, explore directory structure
- Run `xcodebuild` for builds (Release or Debug)
- Run `xcodegen generate` to regenerate the Xcode project
- Run `pre-commit run` or `pre-commit run --all-files`
- Run `git status`, `git log`, `git diff`, `git branch`
- Create and switch to feature branches (`feat/`, `fix/`, `refactor/`, `test/`, `docs/`)
- Stage and commit changes (commits are signed automatically)
- Run `swift build`, `swift test`, `swift package` commands
- Edit source files under `Clippy/`, `Assets.xcassets/`, `Resources/`, `Scripts/`
- Edit `project.yml`, `.gitignore`, `.pre-commit-config.yaml`
- Create or edit documentation files (`*.md`) in the repo
- Run read-only `gh` commands (`gh repo view`, `gh pr list`, `gh issue list`)

## Requires Human Approval (NEVER do autonomously)
- `git push` to any remote (including origin)
- `git merge` into `main`
- Create, merge, or close pull requests (`gh pr create`, `gh pr merge`)
- Delete any branch (`git branch -D`, `git push --delete`)
- Delete files outside the working tree
- Run `rm -rf` on any directory
- Access or modify secrets, `.env` files, credentials, or tokens
- Run `doppler` commands
- Modify CI/CD configuration
- Run `docker` commands that affect running containers
- Run any `sudo` command
- Run `terraform apply` or `terraform destroy`
- Modify files outside `/srv/dev/commercial/Clippy/`
- Modify `~/.gitconfig` or any global configuration
- Send external communications (emails, webhooks, API calls to non-GitHub services)
- Install or remove system packages
- Modify the provision script or health check script
- Access the network except for `gh` CLI read operations and `git fetch`

## File Modification Boundaries
- ALLOWED: Any file under `/srv/dev/commercial/Clippy/` (the repo working tree)
- FORBIDDEN: Anything outside the repo working tree
- FORBIDDEN: `.git/` internals (hooks are managed by pre-commit)
- FORBIDDEN: `.claude/settings.local.json` (managed by the human operator)

## Build Commands
```bash
# Generate Xcode project from project.yml
xcodegen generate

# Build release
xcodebuild -project Clippy.xcodeproj -scheme Clippy -configuration Release build

# Build debug
xcodebuild -project Clippy.xcodeproj -scheme Clippy -configuration Debug build
```

## Pre-commit Hooks
The following hooks run on every commit and must never be bypassed:
- trailing-whitespace, end-of-file-fixer
- check-yaml, check-json
- check-added-large-files (max 1000 KB)
- detect-private-key
- gitleaks (secret scanning)
- protect-main-branch (pre-push: blocks direct push to main)
- audit-log-commit (post-commit: logs to agent audit JSONL)
