# Gmail API connector

Blacki uses Google's Gmail REST API for a connected private Telegram user. It
does not use a shared credential file.

## Configuration

Enable the connector only when the server is ready to store restricted Gmail
data:

```dotenv
GMAIL_ENABLED=true
GMAIL_CLIENT_ID=your-gmail-client-id.apps.googleusercontent.com
GMAIL_CLIENT_SECRET=your-gmail-client-secret
GMAIL_TOKEN_ENCRYPTION_KEY=your-gmail-fernet-key
GMAIL_REDIRECT_URI=https://your-domain.example/integrations/gmail/callback
# Optional download limit, default 25 MiB
GMAIL_MAX_ATTACHMENT_BYTES=26214400
```

`GMAIL_ENABLED` is required and defaults to false. Gmail uses a dedicated OAuth
client, token-encryption key, and callback configuration. `GMAIL_REDIRECT_URI`
is optional and otherwise uses the local callback on port 8080. The Gmail OAuth
client must register the exact callback URL.

Blacki requests only
[`https://www.googleapis.com/auth/gmail.modify`](https://developers.google.com/workspace/gmail/api/auth/scopes).
Refresh tokens are encrypted with the dedicated Gmail Fernet key and stored in
the shared SQLite database. OAuth state is stored as a hash, expires, and can
be consumed only once.

## Telegram flow

1. Send `/connect_gmail` in a private Telegram chat.
2. Authorize the requested Gmail scope in Google's consent screen.
3. Use the Gmail skill after the connection message arrives.
4. Send `/disconnect_gmail` and confirm when access should be revoked.

If Google cannot complete revocation, Blacki disables local Gmail use and
retains the encrypted credential so the disconnect can be retried. It does not
report a successful disconnect until remote revocation succeeds.

Each private Telegram chat has its own connection. Group chats, public agents,
delegated workers, local test identities, and unconnected users receive no
Gmail tools.

## Supported operations

The connector can search non-spam, non-trash messages, read bounded message and
thread bodies, list and read drafts, create drafts and replies, list and create
custom labels, modify custom labels, and download requested attachments into the
current session sandbox. Downloads return only safe metadata and expire with the
sandbox. They are not copied to durable storage or sent through Telegram.
Large message and thread bodies are also materialized in the current session
sandbox when their tool results would otherwise be large. The Gmail tool returns
the sandbox path so the agent can inspect the content incrementally.

Sending requires an explicit Google ADK confirmation. The confirmation includes
the draft ID, recipients, subject, and a content fingerprint returned by the
draft read/create operation. Blacki reloads the draft immediately before
sending and stops if any of those values no longer match.

The connector does not delete messages, access or modify spam and trash, or
change Gmail settings. Downloaded attachments are untrusted files and Blacki
does not automatically open, extract, or execute them. Retrieved email content
is private and may pass through Blacki's configured LLM and conversation storage.

Google's [OAuth token expiration guidance](https://developers.google.com/identity/protocols/oauth2#expiration)
applies while the OAuth application is in testing. Test users may need to
reconnect periodically.
