# Private browser takeover

Browser takeover lets the private Telegram user type passwords and one-time
codes into Agent Browser without sending those values through Telegram messages
or ADK model context.

## How it works

1. The root agent calls `start_browser_takeover` with an HTTPS login page.
2. Blacki starts Agent Browser's interactive stream in the existing session
   sandbox.
3. Blacki sends a single-use link directly to the same private Telegram chat.
   The model receives only the eventual completion status.
4. The link opens a live canvas. Mouse, touch, keyboard, and scroll events pass
   through Blacki to the Agent Browser stream without entering ADK session
   history.
5. The takeover page shows the expected login origin and the current browser
   origin outside the streamed page. If navigation changes origin, the page
   warns the user to verify the new site before typing sensitive data.
6. Clicking **Done** disables the stream and resumes the waiting tool call.
   Disconnects, expiry, replacement, and process shutdown do not count as
   successful human completion.

The URL fragment is exchanged once for an HttpOnly, SameSite cookie. Blacki
stores only token hashes, sandbox routing details, the expected origin, and
completion state in process memory. Restarting Blacki invalidates every
takeover.

## Configure

The sandbox image must contain a compatible `agent-browser` executable. Set:

```dotenv
BROWSER_TAKEOVER_PUBLIC_URL=https://blacki.example.ts.net/browser-takeover
BROWSER_TAKEOVER_TTL_SECONDS=300
BROWSER_TAKEOVER_STREAM_PORT=9223
```

The public URL must end with `/browser-takeover`. HTTPS is mandatory except on
loopback during development. Keep the endpoint on your tailnet and restrict it
with Tailscale ACLs or equivalent device authentication. The reverse proxy must
forward WebSocket upgrades on `/browser-takeover/ws`.

Blacki's production Compose override publishes the application on
`127.0.0.1` by default. Point Caddy or Tailscale Serve at that loopback port;
do not publish the application port on every host interface.

## Operational limits

- One takeover can run per Telegram conversation.
- Links are single-use and expire after five minutes by default.
- Only matching private-chat sender and chat identities can create a takeover.
- A model turn remains paused until the user clicks **Done** or the link expires.
- Starting a new Telegram turn may cancel the waiting model turn.
- Takeover capability state is process-local. Run one application worker and
  one replica, or guarantee sticky routing for the originating tool execution,
  `/redeem`, `/ws`, and `/complete`. A multi-worker or multi-replica deployment
  without shared takeover state will intermittently reject valid links.
- Cookies and local storage created by the target website remain inside the
  session sandbox and are as sensitive as the original password.

Do not enable screenshots, video, HAR files, browser command tracing, or debug
payload logging while a takeover is active. Blacki does not record the input
events, but the target website and Telegram still have their ordinary service
metadata. This feature keeps credentials out of LLM and Blacki content logs; it
does not claim that no system involved in the login has operational logs.
