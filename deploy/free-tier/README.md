# AquaScope free tier

A Cloudflare Worker that lets Explorer visitors with no API key use Claude for a limited time. Anthropic's AI
for Science program pays for it. The Explorer stays a static page, and the key lives here as a Worker secret.

## What it does

| Endpoint | What it does |
|---|---|
| `GET /status` | Whether the tier is open, its end date, the models it serves, the per-visitor allowance, the sponsor line and the Turnstile site key. |
| `POST /session` | Takes a Turnstile token and returns a signed session token valid for 24 hours. The Worker needs no store to recognise it. |
| `POST /run` | Counts the start of an Ask, a map command or a study against the visitor's daily runs. |
| `POST /v1/messages` | The Messages API, with the session token as the `x-api-key`. Each request is checked, charged and passed on to Anthropic. |
| `GET /stats` | Daily totals: dollars, calls, refusals, runs by kind and visitors. Needs `Authorization: Bearer <ADMIN_TOKEN>`. |

A request is passed on only if it is one the Explorer sends:

- the model is Claude Haiku 5.5 or Sonnet 5.5;
- `max_tokens` is at most 16,000;
- the only tools are the Explorer's own (no server tools);
- `output_config` sets effort only;
- the only beta is the server-side fallback;
- the system prompt names AquaScope.

That last check is a speed bump, not a lock. The budgets are what bound misuse. Each call is charged, from the
usage block Anthropic returns, against four budgets:

| Budget | Default | Variable |
|---|---|---|
| The whole tier, per UTC day | $30 | `DAILY_USD` |
| The whole tier, per month | $1,000 | `MONTHLY_USD` |
| One visitor (one session), per day | $2.50 | `VISITOR_USD` |
| One IP address, per day | $8 | `NETWORK_USD` |
| Questions and map commands, per visitor per day | 20 | `VISITOR_ASKS` |
| Studies, per visitor per day | 2 | `VISITOR_STUDIES` |

A spent budget is answered with HTTP 402, which the package's client does not retry. IP addresses are stored only
as a salted daily hash. Prompts and replies pass through and are never stored or logged.

The ledger is a single Durable Object, so every check and charge is serialised. Per-visitor rows are pruned after
a day, and the daily totals are kept.

## Deploy (once)

1. **A key just for this tier.** In the Anthropic Console, create a workspace (for example `aquascope-free-tier`)
   and set a monthly spend limit on it. Then create an API key in that workspace. If the Worker misbehaves, this
   limit is what stops the spending.
2. **Turnstile.** In the Cloudflare dashboard, open Turnstile, add a widget in "Managed" mode, and give it the
   hostnames `rekin226-aquascope-explorer.static.hf.space` and `localhost`. Copy its site key into
   `TURNSTILE_SITE_KEY` in `wrangler.toml`. The site key is public; keep the secret for step 4.
3. **Log in.**

   ```bash
   cd deploy/free-tier
   npm install
   npx wrangler login
   ```

4. **Secrets**, each pasted when asked:

   ```bash
   npx wrangler secret put ANTHROPIC_API_KEY
   npx wrangler secret put TURNSTILE_SECRET
   npx wrangler secret put SESSION_SECRET
   npx wrangler secret put ADMIN_TOKEN
   ```

   `SESSION_SECRET` and `ADMIN_TOKEN` are any long random strings (`openssl rand -hex 32`). If the key spans
   several workspaces, also set `ANTHROPIC_WORKSPACE_ID`.

5. **Deploy.**

   ```bash
   npx wrangler deploy
   curl https://aquascope-free-tier.<your-subdomain>.workers.dev/status
   ```

6. **Point the Explorer at it.** Set `freeTierUrl` in `explorer/config.js` to the Worker's URL and merge. The
   **Free** tier appears in Ask once `/status` says `open` and has a site key.

## Run it

- **Usage.** `curl -H "Authorization: Bearer $ADMIN_TOKEN" https://…workers.dev/stats` returns the last 14 days.
  The stop-rule review at week 6 reads `runs.study`: fewer than 20 a week means switching the tier off.
- **Change a budget or the end date.** Edit `wrangler.toml`, then run `npx wrangler deploy`.
- **Switch it off.** Set `FREE_TIER_ENABLED = "false"` and deploy. Every run endpoint then answers that the tier
  is off, and the Explorer hides the tier. After `END_DATE`, the tier closes the same way without a deploy.

## Develop

```bash
npm test                                   # the rules and the ledger, with node --test
npx wrangler dev --var TURNSTILE_SITE_KEY:1x00000000000000000000AA \
  --var TURNSTILE_SECRET:1x0000000000000000000000000000000AA \
  --var SESSION_SECRET:dev --var ADMIN_TOKEN:dev --var ANTHROPIC_API_KEY:$ANTHROPIC_API_KEY
```

These are Cloudflare's always-pass Turnstile test keys, and any token passes the check with them.
`src/core.js` holds the rules, `src/ledger.js` the budgets, and `src/index.js` the routes and the Durable
Object.
