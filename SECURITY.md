# Security policy

## Supported versions

Only the latest release on `master` gets fixes. Install from `master` (Python package) or
re-run `/plugin marketplace update kaggle-tools` (Claude Code plugin) to pick them up.

## What the plugin does

The Claude Code plugin ships **skills only**: `SKILL.md` instruction files and Markdown
references. It registers no hooks, MCP servers, slash commands or agents, runs no code on
install, and reads no credentials. The `kaggle_tools` Python package is separate and only
runs when you call it; the `kaggle_tools.kernel` helpers shell out to the official `kaggle`
CLI, which uses your own `~/.kaggle` credentials.

## Reporting a vulnerability

Please use GitHub's [private vulnerability reporting](https://github.com/yassineAlouini/kaggle-tools/security/advisories/new)
instead of a public issue. Include the affected file or skill, a reproduction and the
impact. Expect a first answer within a week.
