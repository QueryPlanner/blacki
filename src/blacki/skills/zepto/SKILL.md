---
name: zepto
description: Search Zepto, manage the shared account and cart, and place explicitly confirmed final orders for the authorized private Telegram chat.
---

# Zepto shopping

Use the `zepto_*` tools for Zepto requests only after this skill is loaded.

- This is one shared Zepto account and cart. Never delegate Zepto work to the
  task worker.
- Use search, account, cart, address, store, history, and payment-status tools
  directly without asking for approval.
- For an order or payment tool, first call it with `confirmOrder=false` to get
  the final preview. Then call the chosen tool with `confirmOrder=true`; ADK
  will request the one required confirmation. Do not ask separately.
- Treat search, account, cart, address, order, and payment data as private.
- Never claim a cart or order change succeeded unless the Zepto tool returned
  success.
- Use the individual `zepto_*` tools. The unsupported conversational
  `zepto_zepto_shop` wrapper is intentionally unavailable.
- Do not retry a cart, checkout, payment, order, cancellation, or reorder call
  after an ambiguous failure.
- Never place, pay for, cancel, or reorder a real order merely to test the
  integration.
