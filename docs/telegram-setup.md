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
```

Replace `replace-me` with the token from BotFather. Blacki validates that the
token is present and follows Telegram's `number:string` format at startup.

At least one model provider must also be configured. See
[Configuration](base-infra/environment-variables.md).

### Optional Kokoro speech replies

To let the Telegram-only root agent turn text into playable MP3 audio, add:

```dotenv
KOKORO_TTS_BASE_URL=http://100.x.y.z:8880
KOKORO_TTS_VOICE=af_heart
```

Use the Tailscale IP or MagicDNS name that is reachable from the Blacki
container. `localhost` refers to the Blacki container itself, not a Kokoro
server on another machine. Blacki calls Kokoro's OpenAI-compatible
`/v1/audio/speech` endpoint, keeps the bounded MP3 in memory, and uploads it to
the current Telegram chat or topic. Synthesis and upload are serialized so only
one audio payload is retained at a time. The tool is not registered on the
public ADK HTTP runner or delegated task worker.

When this private tool is configured, Blacki disables content-rich ADK and
OpenInference logging. Tool notifications may show that speech synthesis is
running, but never include the text being spoken. ADK session history still
retains the model's tool call and arguments.

### Optional Connect Google Health

Blacki can read normalized health summaries after a user completes Google OAuth
from a private Telegram chat. This is intentionally named **Connect Google
Health**: Blacki does not request Apple ID credentials, access HealthKit, scrape
Fitbit, or receive arbitrary Apple Health records. The user must first configure
an Apple Health-to-Google Health/Fitbit-compatible import path if their account
and app version support it.

Configure the Google Cloud OAuth web client and the `GOOGLE_HEALTH_*` values in
[Configuration](base-infra/environment-variables.md), then set the callback URL
to the exact public HTTPS URL. In Telegram:

1. Send `/connect_health` in a private chat.
2. Open the one-time Google authorization link and grant only the requested
   read-only categories.
3. Return to Telegram and use `/health_refresh` for an on-demand sync or
   `/health_summary` for the latest stored records.
4. Use `/disconnect_health`, then confirm the button, to revoke the token
   best-effort and delete Blacki's stored token, normalized records, and pending
   OAuth state.

The background sync runs every 12 hours by default and fetches a bounded recent
window so late device imports can replace earlier daily records. Missing values
are omitted rather than guessed. Stored data is limited to normalized daily
activity, workout, sleep, heart-rate, weight, and body-fat summaries; raw
Google payloads and provider IDs are not persisted in the summary table.

Google Health availability does not prove that a particular Apple Health metric
was imported. Test the desired categories on a non-production account before
promising steps, workouts, sleep, or heart-rate coverage to users.

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
| `/connect_health` | Send a Google Health authorization link |
| `/health_refresh` | Fetch recent Google Health data (rate limited) |
| `/health_summary` | Show normalized daily records and trends |
| `/disconnect_health` | Confirm disconnection and local health-data deletion |

Then send a normal message and confirm the model responds. Blacki does not
currently implement a `/clear` command.

If Kokoro speech is configured, ask the bot to “send that as audio” and confirm
Telegram receives an MP3 in its native audio player. Kokoro failures return a
text error and do not create a local audio file.

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

Telegram turns automatically show one live-updating tool status message while
tools run. The status is updated in place and ends with the elapsed working
time, so tool names and arguments are not posted as separate chat messages.

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
