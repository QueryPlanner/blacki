# Telegram setup

Blacki connects to Telegram with long polling. It does not implement webhook
mode, so you do not need a public domain, TLS certificate, or inbound Telegram
port.

## 1. Create a bot

1. Open the verified
   [BotFather](https://core.telegram.org/bots/features#botfather) account in
   Telegram.
2. Send `/newbot`.
3. Choose a display name and a username ending in `bot`.
4. Copy the token.

Treat the token like a password. Anyone with it can control the bot.

## 2. Configure Blacki

In `.env`:

```dotenv
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=replace-me
TELEGRAM_TOOL_NOTIFICATIONS=false
```

Replace `replace-me` with the token from BotFather. Blacki validates that the
token is present and follows Telegram's `number:string` format at startup.

At least one model provider must also be configured. See
[Configuration](base-infra/environment-variables.md).

## 3. Start or recreate the service

Docker Compose:

```bash
docker compose -f compose.yaml -f compose.prod.yaml up --build -d
docker compose -f compose.yaml -f compose.prod.yaml logs --tail=100 agent
```

Local Python:

```bash
uv run python -m blacki.server
```

Look for startup logs confirming that the Telegram configuration was detected
and polling started.

## 4. Verify

Open the bot in Telegram and send:

| Command | Behavior |
| --- | --- |
| `/start` | Show the welcome message |
| `/help` | Show the supported command summary |
| `/model` | Open the model and thinking settings panel |
| `/thinking` | Open the supported reasoning-effort choices for the active model |
| `/reset` | Start a fresh conversation session |

Then send a normal message and confirm the model responds. Blacki does not
currently implement a `/clear` command.

The `/model` panel stores one profile for the Telegram chat. Choose a model,
then select a thinking effort supported by that model. `Default` uses
`ROOT_AGENT_REASONING_EFFORT` when configured and otherwise leaves the
provider's reasoning setting unchanged. `Off` is shown only when the model
advertises that it can disable reasoning. Changes apply to the next turn and
do not change existing conversation history. Changing models resets the
chat's explicit thinking choice to `Default`, avoiding an unsupported setting
being carried to a different model. `/thinking` is a shortcut to the same
capability-aware menu.

To verify image input, select a vision-capable model, send a Telegram photo,
and optionally add a caption as the instruction. Without a caption, Blacki asks
the model to describe the image. Native photo input is limited to 10 MB;
documents, audio, video, and voice messages continue to use the sandbox upload
path. A model that does not support images will return the normal photo
processing error without changing the selected model.

## Tool notifications

`TELEGRAM_TOOL_NOTIFICATIONS=true` sends short tool-use notices for Telegram
turns. It is opt-in because it adds chat traffic and may expose tool names.

## Security

- Never paste the token into an issue, log, command history, or committed file.
- Keep `.env` mode `600`.
- If the token is exposed, revoke and regenerate it through BotFather.
- Group privacy settings are controlled through BotFather; review them before
  adding the bot to a group.

## Troubleshooting

If the bot does not respond:

```bash
docker compose -f compose.yaml -f compose.prod.yaml ps
docker compose -f compose.yaml -f compose.prod.yaml logs --tail=200 agent
```

Check that:

- `TELEGRAM_ENABLED=true`;
- the token contains no quotes or trailing whitespace;
- one real model API key is active;
- the model identifier matches that provider; and
- the VPS can make outbound HTTPS requests.

See [Troubleshooting](troubleshooting.md) for startup and model failures.
