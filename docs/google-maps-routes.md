# Google Maps Routes

Blacki can use the Google Maps Routes API for fresh distance, ETA, traffic, and
route-comparison questions. The integration is optional and remains disabled
unless `GOOGLE_MAPS_ROUTES_API_KEY` is configured.

## Configure the API

1. Enable billing and the Routes API in the Google Cloud project.
2. Create a server-side API key dedicated to this integration.
3. Restrict the key to the Routes API and to the deployed server where
   practical.
4. Configure quotas and billing alerts.
5. Set the key in `.env`:

```dotenv
GOOGLE_MAPS_ROUTES_API_KEY=replace-me
```

Do not reuse the Gemini `GOOGLE_API_KEY`. Separating the keys allows independent
restrictions, rotation, and quotas.

For the repository's production deployment workflow, add the same value as the
GitHub Actions environment secret `GOOGLE_MAPS_ROUTES_API_KEY`. Code-quality
jobs do not need this secret because provider calls are mocked.

## Agent capabilities

`get_route_estimate` returns:

- distance in meters and kilometers;
- traffic-aware and static durations;
- calculated traffic delay;
- optional alternate routes;
- provider fallback and route warnings;
- Google Maps attribution.

`compare_route_scenarios` compares up to five explicitly named scenarios for
the same endpoints. A scenario can vary departure time, travel mode, traffic
model, and avoid options. Requests run with bounded concurrency to limit burst
traffic and cost.

For a current driving estimate, the agent uses:

- travel mode `DRIVE`;
- departure time `now`; and
- traffic model `BEST_GUESS`.

`OPTIMISTIC` and `PESSIMISTIC` are also supported for driving. Non-driving
modes use `NONE` because Google traffic models are limited to driving routes.

## Saved routes and scheduled updates

When SQLite storage is enabled, Blacki can save common routes and schedule
recurring traffic checks. It persists only Google place IDs, user-authored
labels, route preferences, and reminder metadata. Raw addresses, API responses,
and traffic snapshots are not stored.

Saved-route operations are owner-scoped. In Telegram they are intentionally
limited to private chats because group and topic sessions use a shared chat
identity. `GOOGLE_MAPS_SAVED_ROUTE_LIMIT` defaults to 20 saved routes per user,
and `GOOGLE_MAPS_ROUTE_UPDATE_LIMIT` defaults to 10 active traffic updates.
Scheduled checks must be at least 15 minutes apart.

## Location and time inputs

Plain location strings are sent as addresses. A known Google place ID can be
supplied using the `place_id:` prefix:

```text
place_id:ChIJ...
```

Future departure times must be RFC 3339 timestamps containing a timezone
offset. This keeps an instruction such as "8:30" from being interpreted in the
wrong timezone.

## Operational boundaries

- Route responses are point-in-time estimates. The Routes API does not provide
  continuous tracking or a traffic push subscription.
- Avoid-toll, highway, and ferry options are preferences, not guarantees.
- Walking, bicycling, and two-wheeler results are beta and include a warning.
- The integration requests a fixed minimal response field mask. It does not
  request toll pricing, eco routes, traffic-colored polylines, or route
  matrices.
- Route responses and traffic snapshots are not persisted. Google Maps
  Platform storage and attribution policies still apply to downstream uses.
- Provider errors are normalized without logging the API key, request payload,
  exact locations, or resolved place IDs. When Routes is enabled, OpenInference
  input and output capture is disabled for the process.

See the official [Compute Routes
reference](https://developers.google.com/maps/documentation/routes/reference/rest/v2/TopLevel/computeRoutes),
[traffic model
guide](https://developers.google.com/maps/documentation/routes/traffic-model),
[field-mask
guidance](https://developers.google.com/maps/documentation/routes/choose_fields),
and [Routes
policies](https://developers.google.com/maps/documentation/routes/policies).
