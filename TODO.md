# TODO – Modal ComfyUI Improvements

## P0 – Critical
- Replace mutable list defaults in Pydantic models with `Field(default_factory=list)`.
- Separate snapshot vs GPU initialization; keep torch/CUDA imports out of `@modal.enter(snap=True)` and verify CUDA in `snap=False` path.
- Harden ComfyUI startup: context-manage subprocess, stream logs, add liveness watcher, extend wait with backoff.
- Persist `prompt_id`, node counts, current node, and timestamps in job data and API responses.
- Implement reliable progress tracking using ordered node list + WebSocket (`execution_start`/`executed`/`progress`) with history fallback and throttled writes.
- Create per-job input directories, update media handling, and clean them post-run.
- Enforce real cancellation by calling Comfy interrupt/queue cancel endpoints and updating job status appropriately.
- fix the 'ComfyUI progress is never visible to the caller'
## P1 – High
- Add health probe before each job (`/system_stats`) and surface failure early.
- Reduce volume commit frequency; throttle progress/status writes.
- Capture tail of Comfy stdout/stderr on failure and persist in `error_log_tail`.
- Improve WebSocket resilience: timeouts, ping/keepalive, retry/backoff logic.
- Increase media validation for URLs/content types alongside existing size checks.
- Retry wrappers for outbound HTTP (Comfy, webhooks) with exponential backoff.

## P2 – Medium
- Stream large outputs with `stream=True` and chunked reads when pulling from Comfy.
- Pin external custom node/model installs to commits/tags and relax clone timeouts.
- Optional: add rate limiting + deep `/health` route on FastAPI app.
- Optional: revisit memory snapshot usage (either complete helper setup or disable).
- Optional: switch Comfy launch to `comfy launch --background ...` for cleaner lifecycle.

## Test Follow-Up
- Update `test_wan22_api.py` expectations for new job metadata, progress semantics, and cancellation behavior.
- Add tests for per-job input directories and invalid media handling.
- Mock WS/history responses to cover retry/backoff and throttled writes.
- Verify health-check failures and cancellation propagate correct status codes.
