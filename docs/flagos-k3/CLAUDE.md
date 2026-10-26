# FlagOS SpacemiT K3 - Claude Code rules

Claude Code loads this file and `PROJECT.md` at every session, through the two imports in `.claude/CLAUDE.md`. `PROJECT.md` holds the project context and the working rules; `AGENTS.md` beside it is the Codex guide; this file holds what only Claude Code uses. The `coder` subagent is defined in `.claude/agents/coder.md`.

## Upstream instruction files

- The repository root's `CLAUDE.md` and `AGENTS.md` stay upstream llama.cpp's, byte for byte (user, 2026-10-26). Fork instructions go in this folder: this file, `PROJECT.md`, or `AGENTS.md` (Codex). `.claude/settings.json` denies Claude Code's edits to the two root files.
- Check: `git diff --stat ba360ef -- AGENTS.md CLAUDE.md` prints nothing.

## Workflow
- Plan and reason yourself. For implementation, delegate to the `coder` subagent with a precise spec: files, interfaces, edge cases, and how to test.
- After it returns, verify yourself: read the diff, run the tests, and send it back with specific fixes if needed.
- Do trivial one-line edits directly; don't delegate them.
