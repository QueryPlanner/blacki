---
name: zepto
description: Search Zepto, inspect the shared account and cart, and perform explicitly confirmed Zepto shopping actions for the authorized private Telegram chat.
---

# Zepto shopping

Use the `zepto_*` tools for Zepto requests only after this skill is loaded.

- This is one shared Zepto account and cart. Never delegate Zepto work to the
  task worker.
- Every Zepto tool call requires a user confirmation. Present the exact tool
  name and arguments, then wait for the confirmation flow.
- Treat search, account, cart, address, order, and payment data as private.
- Never claim a cart or order change succeeded unless the Zepto tool returned
  success.
- Prefer the individual `zepto_*` tools. Do not call `zepto_zepto_shop`; Zepto
  documents that conversational wrapper as unsuitable for MCP clients.
- Do not retry a cart, checkout, payment, order, cancellation, or reorder call
  after an ambiguous failure.
- Never place, pay for, cancel, or reorder a real order merely to test the
  integration.
