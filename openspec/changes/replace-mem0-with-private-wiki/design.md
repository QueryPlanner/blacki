## Decisions

- The wiki is stored beside the application SQLite database. Relative Markdown paths describe folders. The server renders pages from claim rows, so updating a claim also updates the next page read and search.
- The verified Telegram sender owns the wiki across topics. The opt-in, pause, and forget controls live under `/memory`. Writes require verified user evidence and a current revision. Ambiguous conflicts stay disputed.
- History search opens the Telegram ADK SQLite database read-only. New events carry server-stamped origin metadata. Eligible legacy sessions can be searched as context. Source snippets and tool arguments are kept out of stored memory tool events.
- The private system policy distinguishes original messages from derived wiki pages. It teaches lookup before upsert, cited recall, history fallback, and bounded review.
- Requests beginning with "Schedule wiki review" bind a reminder to the verified owner. Delivery checks access again and grants only the three memory tools. The reminder itself supplies no factual evidence.
- Keep legacy Mem0 records outside live recall. Migration and deletion require separate authorization.
