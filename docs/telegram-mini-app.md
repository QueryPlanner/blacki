# Telegram Mini App

The current Mini App is a local, fixture-only UI preview. It lets you compare
the Telegram companion mode with the in-app assistant and try the rounded
Blacki Playroom theme. It does not read or write your Blacki database, call a
model, connect to Google Health, access R2, or schedule Telegram messages.

## Run the local preview

Install the pinned front-end build tools and generate both static stylesheets:

```bash
npm ci
npm run build
```

Enable only the demo route, then start Blacki:

```bash
MINIAPP_DEMO_ENABLED=true uv run server
```

Open <http://127.0.0.1:8080/miniapp>. If you run Blacki on a different port,
use that local port instead. The host defaults to `127.0.0.1`.

## Demo behavior

- Switch between **Telegram companion** and **In-app assistant** from the
  lavender mode card. Both show the same in-memory fixture records.
- Try the meal correction, three-set workout, reminder reschedule, and chart
  filter tasks in either mode.
- Slider and exact-number controls stay synchronized. Changes remain drafts
  until you save. Switching modes offers to keep, save, or discard a draft.
- Use **Reset demo** to restore the starting records. A page refresh also
  starts with the fixtures again.
- The model, memory, file, tracker, and activity screens are representative
  demo panels. Provider status is labelled unavailable where it is not
  connected.

`MINIAPP_DEMO_ENABLED` defaults to `false`. `MINIAPP_ENABLED` also defaults to
`false`; it controls the future authenticated live experience and does not
turn on fixture mode. A live Telegram trial needs verified launch data and a
reachable HTTPS host. Do not expose the local demo as a production service.

## Comparison worksheet

Use the same starting records in both modes. Reset before repeating a task in
the reverse order. Record what you actually notice in
[the comparison worksheet](mini-app-comparison.md).
