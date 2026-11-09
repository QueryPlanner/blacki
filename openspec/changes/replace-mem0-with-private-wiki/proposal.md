## Why

Blacki needs durable, sourced memory and a way to find prior conversations after a reset.

## What Changes

- Replace Mem0 with a private SQLite wiki enabled by default.
- Add `wiki_read`, `wiki_write`, and `search_past_sessions` to verified private Telegram turns.
- Teach Blacki to ingest facts, answer with sources, and review the wiki using [Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).
- Use the existing reminder delivery path for owner-authorized reviews. A dedicated dreaming scheduler is outside this change.

## Capabilities

### New Capabilities

- `private-user-wiki` supports sourced claims, private history retrieval, and owner-directed review.

### Modified Capabilities

## Impact

Telegram, ADK tools and system policy, host SQLite, reminder authorization, and Mem0 dependency removal.
