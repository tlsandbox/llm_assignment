# Troubleshooting Guide

## 1) Browser shows `{\"detail\":\"Not Found\"}` at `127.0.0.1:8000`

Cause:

- Another local service is bound to port 8000.

Fix:

1. Run the project with:
   ```bash
   ./scripts/run_api_dev.sh
   ```
2. Open the exact URL printed by the script (default `http://127.0.0.1:8001`).
3. Do not assume port 8000 is your app.

## 2) UI shows `Failed to fetch`

Cause:

- API and frontend can be on different ports when local services conflict.

Fix:

1. Start app from one process using `./scripts/run_api_dev.sh`.
2. Use only the printed origin in browser.
3. Hard refresh (`Cmd+Shift+R`) after restarting.

## 3) AI requests timeout at 45s

Cause:

- Upstream OpenAI request delay or network instability.

Fix:

1. Verify health:
   ```bash
   curl http://127.0.0.1:8001/api/health
   ```
2. Confirm `OPENAI_API_KEY` is set in `.env`.
3. Tune timeouts in `.env`:
   - `RN_AI_SEARCH_TIMEOUT_SECONDS`
   - `RN_AI_IMAGE_TIMEOUT_SECONDS`
   - `RN_AI_MATCH_TIMEOUT_SECONDS`
4. Retry. Fallback paths should still return usable results.

## 4) Personalized page says no session

Cause:

- Page opened directly without running search/image flow.

Fix:

1. Go to Home (`/`).
2. Run a text search or image upload.
3. Let app route to `/personalized?session=<id>`.

## 5) Match explanation seems too generic

Fix:

- Click `Check Your Match` to open modal details.
- Review per-signal rows for:
  - Expected vs Product values
  - Matched cues
  - Missing cues
  - Judgement sentence
