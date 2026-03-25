# Bugs & Fixes — Discovery Flow Debugging Session (2026-03-23)

## Fixed

### 1. Zero logging across the codebase
- **Problem:** No logging in any agent or service file — impossible to debug issues.
- **Fix:** Added `logging.getLogger("wandermust.*")` to all files in `agents/` and `services/`. Logs LLM calls, API requests/responses, result counts, and errors.
- **Files changed:** `llm_helper.py`, `serpapi_client.py`, `tavily_client.py`, `reddit_client.py`, `weather_client.py`, `geocoding.py`, `social.py`, `flights.py`, `hotels.py`, `weather.py`, `synthesizer.py`, `onboarding.py`, `discovery_chat.py`, `suggestion_generator.py`, `app.py`

### 2. LLM JSON responses wrapped in markdown code fences
- **Problem:** GPT-4o wraps JSON output in `` ```json ... ``` `` code fences. All agents did raw `json.loads(response.content)` which failed silently, falling back to empty results or mock data.
- **Fix:** Added `parse_json_response()` helper in `llm_helper.py` that strips code fences before parsing. Replaced `json.loads(response.content)` with `parse_json_response(response.content)` in `discovery_chat.py`, `suggestion_generator.py`, `onboarding.py`, `social.py`.
- **Verified:** Tested end-to-end — `extract_trip_intent` and `generate_suggestions` now return correct data.

## Open Bugs (Still Needs Fixing)

### 3. `_finish_discovery()` infinite rerun loop (CRITICAL)
- **Problem:** When the last discovery question is answered, `_handle_answer()` calls `_finish_discovery()` directly, then calls `st.rerun()`. But `_finish_discovery()` contains slow LLM calls (trip intent extraction + suggestion generation, ~10s total). During this time, Streamlit reruns the script repeatedly, and since the phase hasn't been persisted as "done" yet, `_finish_discovery` gets triggered again and again — creating an infinite loop of LLM calls.
- **Root cause:** `_handle_answer` calls `_finish_discovery` synchronously before `st.rerun()`. The phase is set to `"loading_suggestions"` at the start of `_finish_discovery`, but Streamlit's button callback mechanism causes the entire script to re-execute, and the `_handle_answer` callback fires again because the button state hasn't been cleared.
- **Attempted fix:** Set `discovery_phase = "loading_suggestions"` at the start of `_finish_discovery` to guard against re-entry. This didn't fully work because the function is called from within a button callback, not from the phase-based if/elif chain.
- **Proper fix needed:** Don't call `_finish_discovery()` from `_handle_answer()`. Instead, just set `discovery_phase = "loading_suggestions"` and `st.rerun()`. Then in the main phase if/elif chain, add a handler for `"loading_suggestions"` that calls the LLM logic. This way it only runs once per rerun and the phase gate prevents re-entry. Same pattern needed for `_finish_onboarding()`.

### 4. Duplicate `logging.FileHandler` on every Streamlit rerun
- **Problem:** `app.py` adds a `FileHandler` to the root logger at module level. Streamlit re-executes the entire script on every interaction, so handlers accumulate — each rerun adds another handler, causing duplicate log lines (N handlers after N reruns).
- **Fix needed:** Guard the handler addition with a check, e.g.:
  ```python
  if not any(isinstance(h, logging.FileHandler) for h in logging.getLogger().handlers):
      # add handler
  ```

### 5. Old Streamlit process on default port still running
- **Problem:** There's an old `streamlit run app.py` process (PID 29860, started at 8:57PM) running on the default port alongside our port 8501 instance. Could cause confusion.
- **Fix needed:** Kill it manually: `kill 29860`

## Notes
- Reddit credentials are placeholder (`your_reddit_client_id`) — Reddit search is skipped silently, only Tavily provides social data.
- The "Optimize When" flow was not tested in this session — only "Discover Where".
- Log file location: `/tmp/wandermust.log`
