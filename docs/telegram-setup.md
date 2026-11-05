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
# Optional: restrict new private chats with one shared access code.
# TELEGRAM_ACCESS_CODE=replace-with-a-dedicated-high-entropy-code
```

Replace `replace-me` with the token from BotFather. Blacki validates that the
token is present and follows Telegram's `number:string` format at startup.

When `TELEGRAM_ACCESS_CODE` is set, new users must send `/start <access-code>`
in a private chat before Blacki processes their messages. The bot consumes this
command locally and attempts to delete it, but Telegram may retain it in
server-side history or backups, so use a dedicated code rather than a password
you use elsewhere. Historical private chats with persisted Blacki sessions are
grandfathered; group chats and forum topics are rejected. Changing the access
code requires passphrase-authorized users to authenticate again but does not
delete any chat history, preferences, reminders, files, or health data.

At least one model provider must also be configured. See
[Configuration](base-infra/environment-variables.md).

### Optional voice-note transcription

To let Blacki turn Telegram's native voice-note button into a normal text
conversation, add a Cloudflare Workers AI account ID and API token:

```dotenv
CLOUDFLARE_ACCOUNT_ID=replace-me
CLOUDFLARE_API_TOKEN=replace-me
```

The token should have Workers AI Read and Write permissions. Blacki sends
voice notes to Cloudflare's hosted
`@cf/openai/whisper-large-v3-turbo` model, then passes the transcript through
the existing Telegram text-turn path so conversation history and tool
confirmations continue to work. Voice notes are kept transiently in memory;
regular audio files continue to use the existing file-upload behavior. If the
credentials are absent, Blacki reports that voice transcription is not
configured instead of failing startup. The raw voice bytes are transient; the
resulting transcript is handled like ordinary Telegram text and follows the
existing conversation-history and logging/privacy settings.

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
public ADK HTTP runner.

When this private tool is configured, Blacki disables content-rich ADK and
OpenInference logging. Tool notifications may show that speech synthesis is
running, but never include the text being spoken. ADK session history still
retains the model's tool call and arguments.

### Optional Connect Google Health

Blacki can read normalized health summaries after a user completes Google OAuth
from a private Telegram chat. If the user grants both nutrition permissions,
Blacki also queues existing meals from that private chat once for the connected
Google account, then exports future meal logs, edits, and deletions. This is
intentionally named **Connect Google Health**: Blacki does not
request Apple ID credentials, access HealthKit, scrape Fitbit, or receive
arbitrary Apple Health records. The user must first configure an Apple
Health-to-Google Health/Fitbit-compatible import path if their account and app
version support it.

Configure the Google Cloud OAuth web client and the `GOOGLE_HEALTH_*` values in
[Configuration](base-infra/environment-variables.md), then set the callback URL
to the exact public HTTPS URL. In Telegram:

1. Send `/connect_health` in a private chat.
2. Open the one-time Google authorization link. Grant the read-only categories
   for summaries. Grant both `googlehealth.nutrition.readonly` and
   `googlehealth.nutrition.writeonly` if you want future meal export. Existing
   connections must reconnect to add these nutrition permissions. The nutrition
   read permission also imports nutrition and hydration history. Reconnect
   existing accounts to grant profile, settings, ECG, irregular rhythm, and
   location read permissions for complete archival ingestion.
3. Return to Telegram and use `/health_refresh` for an on-demand sync or
   `/health_summary` for the latest stored records.
4. Log meals normally. Existing eligible meals are queued once after the
   connection is saved. New meals continue through the background export
   worker. A local Blacki save remains successful when remote sync is pending
   or fails, and the meal must not be logged again. Ask Blacki for meal sync
   status when you want to check the queue, or ask it to retry failed exports.
5. Use `/disconnect_health`, then confirm the button, to revoke the token
   best-effort, cancel pending meal sync, and remove Blacki's stored token and
   normalized health summaries, imported records, and import checkpoints.
   Local calorie logs remain. Blacki does not
   delete records already sent to Google Health, and requests already submitted
   may still finish.

The background sync runs every 12 hours by default and fetches a bounded recent
window so late device imports can replace earlier daily records. Missing values
are omitted rather than guessed. A separate archive worker runs every minute,
committing at most eight bounded pages across accounts and data types per run.
It imports all readable catalog types, including beta skin-temperature sensors,
profile/settings, paired devices, nutrition catalogs, and exercise TCX media.
Existing connections must reconnect after seeing the archival retention
disclosure. A persisted consent version gates the importer independently of
OAuth scopes; authorization flows started before this disclosure keep the old
version and do not enable archival ingestion.

Full JSON payloads, provider identifiers, and unknown fields stay in private
host SQLite. Each response is capped at 8 MiB; oversized pages or media are
reported as incomplete rather than silently truncated. Large or malformed TCX
exports do not block other exercise JSON pages.

Listing imports all available history without a date filter. Floors use
reconciliation because Google does not expose their original records. Total
calories and calories in heart rate zones use minute rollups in one-day ranges,
working backward to the earlier of profile membership and the earliest known
recorded observation. Older observations extend this bound monotonically,
including across rescans and reauthorization. Previously completed aggregate
streams resume only the newly uncovered interval. This remains an inferred
boundary, not a guarantee that aggregate-only data cannot exist earlier.
Set `GOOGLE_HEALTH_ARCHIVE_START_DATE=YYYY-MM-DD` to extend the boundary earlier
or supply a missing membership date. A later override never shortens history.
The archive status reports the actual date range. Empty days do not stop it.

`/health_summary` reads the existing daily summaries and reports separate archive
progress. The agent receives coverage metadata, never full archival payloads.
A completed scan repeats after 24 hours. Observed versions are deduplicated and
retained until disconnect or account replacement. This is an observation
archive: provider deletions do not remove earlier observations. Unreadable
write-only types are listed explicitly; permission gaps and failures prevent a
complete status. Archive payloads never enter R2, the sandbox, or Telegram.

Meal exports retain their separate retry state and one-time local-meal backfill.
Run one scheduler process per `tools.db` to avoid duplicate network work.
Checkpoint comparison and connection generations reject stale concurrent writes.

Google's v4 discovery document currently lists `nutrition-log` as a supported
data type. Blacki writes only the local meal description, kcal, available
macros, meal type, and selected local date; it omits unknown nutrients and does
not substitute food-database estimates. See Google's [nutrition data type]
and [data point REST reference] for the provider contract.

[nutrition data type]: https://developers.google.com/health/data-types/nutrition
[data point REST reference]: https://developers.google.com/health/reference/rest/v4/users.dataTypes.dataPoints

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
| `/model <author/model-id>` | Use an exact OpenRouter model ID for the next turn |
| `/model default` | Restore the configured deployment model |
| `/model` or `/models` | Show the active model, command usage, and thinking settings |
| `/thinking` | Open the supported reasoning-effort choices for the active model |
| `/reset` | Start a fresh conversation session |
| `/connect_health` | Send a Google Health authorization link and consent for optional meal sync |
| `/health_refresh` | Fetch recent Google Health data (rate limited) |
| `/health_summary` | Show normalized daily records and trends |
| `/disconnect_health` | Confirm disconnection and cancellation of meal sync |

Then send a normal message and confirm the model responds. Blacki does not
currently implement a `/clear` command.

If Kokoro speech is configured, ask the bot to “send that as audio” and confirm
Telegram receives an MP3 in its native audio player. Kokoro failures return a
text error and do not create a local audio file.

Send `/model author/model-id` with the exact ID from
[OpenRouter](https://openrouter.ai/docs/api_reference/overview), for example
`/model openai/gpt-5.2`. The optional `openrouter/` transport prefix is also
accepted, as in `/model openrouter/openai/gpt-5.2`. Variant suffixes such as
`:free` are preserved. There is no preset model list. `/model` and `/models`
show the complete active ID and the thinking and reset controls.

Blacki checks the ID's syntax and saves the selection; OpenRouter checks model
availability and account access when the next message runs. Custom selections
require `OPENROUTER_API_KEY` on the Blacki host. They reuse that configured
account and may change model costs, context limits, and tool or image support.

`/model default` clears the chat override. The default remains `ROOT_AGENT_MODEL`,
which the deployment workflow takes from the GitHub Actions secret of the same
name. Reset settings also clears the model and explicit thinking choice.
Neither command writes a new deployment default.

Blacki stores one profile for the Telegram chat, shared across its topics.
After selecting a model, choose a thinking effort supported by that model.
The thinking option `Default` uses
`ROOT_AGENT_REASONING_EFFORT` when configured and otherwise leaves the
provider's reasoning setting unchanged. `Off` is shown only when the model
advertises that it can disable reasoning. Changes apply to the next turn and
do not change existing conversation history. Changing models resets the
chat's explicit thinking choice to `Default`, avoiding an unsupported setting
being carried to a different model. `/thinking` is a shortcut to the same
capability-aware menu.

To verify image input, select a vision-capable model, send a Telegram photo,
and optionally add a caption as the instruction. Without a caption, Blacki asks
the model to describe the image. Native photo input is limited to 10 MB.
Telegram documents whose MIME type is `image/*` also use native inline image
input, are limited to 10 MB, and never need an OpenSandbox instance. Other
documents, audio, video, and voice messages continue to use the sandbox upload
path. Native photos and image documents are still best-effort copied to the
private R2 file catalog when it is enabled; R2 or sandbox availability does not
block their inline model turn. A media group produces one turn with separately
labelled image parts ordered by Telegram message ID, so the model analyzes each
image independently rather than treating the group as one collage. The
`sandbox_view_image` tool remains available for images already in the sandbox
or restored from R2. Because Telegram does not send a separate album-complete
update, Blacki waits for one second of quiet after the latest album item, with
a 15-second cap for stalled delivery. An unusually delayed item after that
quiet window is treated as a separate media group.

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

Health authorization requests only Google Health scopes and disables merging
previous Google grants. A shared OAuth client must not pull Gmail or YouTube
permissions into a Health token. If Google reports `DISALLOWED_OAUTH_SCOPES`,
start a fresh connection with the updated link and verify the OAuth client's
configured Health scopes.

### Query archived Health data

The private Telegram agent can call `query_health_data` for actual archived
records, even when recent normalized summaries are empty. Questions such as
"Show yesterday's sleep and resting heart rate" use the archive directly.
Historical backfill does not need to finish before available records can be read.

Each structured query selects one readable data type and an inclusive date range
of at most 31 days. The default is the latest seven days. The agent can select
fields, paginate through records, or use count, sum, average, minimum, and maximum
for documented numeric metrics. Physical timestamp dates use UTC. Daily and
civil-only records retain their provider calendar date. Sleep uses its end date;
other intervals use their start date. Profile and catalog queries accept no dates.

The query runs against an owner- and account-scoped read-only host connection.
The full archive stays on the host. Only selected results, up to 100 records and
32 KiB per call, enter the model and private conversation history. The tool does
not execute model-written SQL or Python, copy files into a sandbox, upload to R2,
or send a file through Telegram. Private logging filters redact query payloads.

Queries select the latest observed version per provider record ID. Anonymous
payloads remain separate observations. Sums and averages over anonymous inputs
require the explicit `observations` basis because sources or versions can overlap.
Responses report this limitation and acquisition status. Missing values and
booleans do not become zero. Provider-side deletion does not remove archived
observations. Pagination is a live view, so new ingestion can change later pages.
Queries have a two-second execution budget; large archives can hit that limit
even for narrow dates because version selection precedes date filtering.
