# Zepto MCP

Blacki connects to Zepto's hosted
[MCP server](https://github.com/zeptonow/mcp) through the supported
[`mcp-remote`](https://github.com/geelen/mcp-remote) bridge. Zepto rejects
Blacki's generic dynamic OAuth registration with HTTP 403 during validation,
so Blacki uses the same standards-based bridge path documented for other MCP
clients.

## Security model

- One private bridge directory represents one Zepto account and shared cart.
- Only explicitly allowlisted positive private Telegram chat IDs can connect.
  Groups, topics, task workers, and arbitrary HTTP user IDs are rejected.
- Zepto is loaded into a Telegram-only ADK runner. The unauthenticated ADK HTTP
  runner never receives the Zepto skill or toolset, even if a caller spoofs an
  allowlisted Telegram user ID.
- Only final order or payment calls with `confirmOrder=true` require ADK
  confirmation. Reads, cart changes, address changes, and order previews run
  without approval.
- Credentials are plaintext protected by `0700` directories and `0600` files.
- ADK session history retains shopping prompts, tool arguments, and results.
  The configured model also receives tool data needed to answer the request.
- Secure Zepto mode disables content-rich ADK tracing and OpenInference ADK
  instrumentation, removes the ADK content logging plugin, and suppresses
  user/model/tool payloads from Blacki's lifecycle and Telegram preview logs.
  It does not erase session history.

## Authenticate locally

From the repository root, leave `ZEPTO_MCP_ENABLED=false` and run:

```bash
uv run python -m blacki.zepto.auth login
```

Blacki starts the exact locked `mcp-remote` bridge, which opens Zepto's
mobile-number and OTP flow in the browser. The bridge requests Zepto's
`tools:read` MCP scope with PKCE. That scope authorizes access to the tool
server; Blacki separately confirms only final order or payment placement. The
command then lists the complete tool manifest without calling any shopping
tool.

The default credential directory is:

```text
data/credentials/zepto-mcp-remote/
```

The existing `./data:/app/data` Compose mount makes that same directory
available to the container. Do not copy its contents into `.env`, logs, chat,
or source control.

Verify the stored credentials without opening a browser:

```bash
uv run python -m blacki.zepto.auth status
uv run python -m blacki.zepto.auth probe
```

If Zepto access is revoked or you intentionally want to switch accounts, use:

```bash
uv run python -m blacki.zepto.auth login --force
```

The force option deletes only the four known Zepto bridge files inside the
dedicated credential directory, then starts a fresh browser login.

## Enable the root-agent skill

Set the positive numeric ID of the private Telegram chat:

```dotenv
ZEPTO_MCP_ENABLED=true
ZEPTO_MCP_ALLOWED_TELEGRAM_CHAT_IDS=123456789
ZEPTO_MCP_CONFIG_DIR=data/credentials/zepto-mcp-remote
```

Restart Blacki, then ask it to search Zepto or inspect the cart. The first
request loads the Zepto skill and non-final tools run directly. When a final
order or payment call is ready, Blacki shows its exact tool name and arguments;
reply exactly `yes` or `no`.

## Verification boundary

Tool discovery and read calls are safe local verification. Order and payment
tools can generate a preview with `confirmOrder=false`; `confirmOrder=true`
stops at the confirmation interrupt before the real final action. Cart,
address, store, and profile mutations execute without approval and affect the
shared account immediately. Never place, pay for, cancel, or reorder a real
order only to test this integration.

## Runtime and deployment notes

The production image includes Node 22 and installs `mcp-remote` 0.1.38 from the
committed npm lockfile. It never downloads npm packages at runtime. Local
development falls back to the same pinned package through `npx` when no global
bridge exists.

Blacki is intentionally single-process and single-replica when Zepto is
enabled: all allowlisted chats share one account, cart, bridge session, and
credential store. If credentials are revoked, the runtime connection fails
within 15 seconds. Run the login command locally again; do not try to complete
OAuth inside the long-running server.
