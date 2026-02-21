# Modal ComfyUI SaaS - Full Validation Report (2026-02-20)

## Scope
This report validates the current `main` branch state for SaaS readiness (2k+ daily active users) using:
- `README.md`
- `GUIAS.md`
- Runtime behavior on deployed Modal app
- End-to-end API tests against production URL

Validated app URL:
- `https://cezarsaint--comfyui-saas-api.modal.run`

## What was executed

1. Secret + deploy sync
- Updated `comfyui-api-secret` with the provided API key (redacted in this report).
- Redeployed with `.venv/bin/modal deploy comfyui_api.py`.
- Deployment succeeded.

2. Runtime verification
- Health endpoint: `200 OK`.
- Auth verification with correct key: protected endpoint returned `404 Job not found` (meaning auth passed).
- Invalid/missing auth: correctly rejected with `401`.

3. Model/cache verification
- Ran `.venv/bin/modal run comfyui_api.py::verify_setup`.
- Result: `Models: 2 files, 13.1 GB`.
- Result: `Custom nodes: 1 (memory_snapshot_helper)`.

4. End-to-end generation
- Ran `test_run.py` against deployed API.
- Job completed successfully.
- Output uploaded to R2 and returned as signed URL.

5. E2E suite checks
- `test_e2e.py --test health`: pass.
- `test_e2e.py --test auth`: pass.
- `test_e2e.py --test e2e`: pass.
- `test_e2e.py --test cancel`: pass.
- `test_e2e.py --test quota`: pass.

6. Concurrency check (3 simultaneous requests)
- Submitted 3 jobs in parallel.
- All 3 completed.
- Images were downloaded and saved to `tests_outputs/`.
- JSON metrics saved to `tests_outputs/concurrency3_validation_20260220_015730.json`.

## Measured results

### Single-run execution
From `test_run.py`:
- Job ID: `7f5daf4a-6bc9-401e-89e5-83d3e3137f6f`
- Total time: `46.7s`
- Output uploaded to R2: yes (`ComfyUI_00066_.png`)

### E2E fast run
From `test_e2e.py --test e2e`:
- Job ID: `73679763-1de3-41b7-a16d-0bba79f80403`
- Completed in ~`7.8s`
- Output URL returned from R2: yes

### Cancellation
From `test_e2e.py --test cancel`:
- Job moved `queued/running -> cancelled`
- API returned `Job cancellation requested`
- Final status confirmed: `cancelled`

### Quota
From `test_e2e.py --test quota`:
- 8 submissions in burst
- Accepted: 5
- Rejected with 429: 3
- Behavior matches `MAX_ACTIVE_JOBS_PER_USER=5`

### 3x concurrency metrics
From `tests_outputs/concurrency3_validation_20260220_015730.json`:
- `concurrency3_run1`: queue `1.07s`, run `6.8s`, total `7.87s`
- `concurrency3_run2`: queue `9.01s`, run `17.49s`, total `26.51s`
- `concurrency3_run3`: queue `9.40s`, run `14.97s`, total `24.37s`
- All outputs uploaded to R2 and persisted locally in `tests_outputs/`

## Cost/containers behavior
- During burst tests, multiple containers appeared active.
- After waiting for scale-down window, active containers dropped back to a single long-lived container.
- This is consistent with `GPU_MIN_CONTAINERS=0` behavior and periodic API activity.

## Code review findings (critical for production hardening)

### 1) `.env.example` is currently cost-dangerous and inconsistent
File: `.env.example:12`
- Example defaults currently suggest:
  - `GPU_MAX_CONTAINERS=20`
  - `GPU_MIN_CONTAINERS=1`
  - `API_MAX_CONTAINERS=10`
  - `QUEUED_TIMEOUT_SECONDS=240`
- Runtime code defaults are much more conservative (`comfyui_api.py:44-51`).

Why this matters:
- New operators following `.env.example` can accidentally over-provision and burn credits.

### 2) Ownership isolation is not enforced
Files: `comfyui_api.py:786`, `comfyui_api.py:822`
- `GET /v1/jobs/{job_id}` returns any job if caller has global API key.
- `GET /v1/jobs` allows filtering by `user_id`, but caller can query arbitrary values.
- `user_id` is optional at job creation (`comfyui_api.py:217`, `comfyui_api.py:737`).

Why this matters:
- In multi-tenant SaaS, this is a data-isolation risk.

### 3) SSRF protection is only basic
File: `comfyui_api.py:278`
- Validation blocks obvious localhost/private-IP literals.
- It does not enforce DNS resolution checks across redirects and post-resolution private IP checks.

Why this matters:
- Advanced SSRF bypasses are still possible via DNS/redirect tricks.

### 4) Cancellation is functional but not “strong interrupt”
Files: `comfyui_api.py:858`, `comfyui_api.py:463`
- API marks cancelled + calls `FunctionCall.cancel()`.
- Worker checks cancellation every 30s and stops loop.
- No explicit Comfy queue interruption endpoint call in worker (`/interrupt`, `/queue` cancellation).

Why this matters:
- Some in-flight graph executions may continue briefly before full stop.

### 5) Webhooks are unsigned
File: `comfyui_api.py:658`
- `_send_webhook` posts plain JSON without request signature header.

Why this matters:
- Receiver cannot cryptographically verify sender authenticity.

## Alignment with README/GUIAS

Aligned and working now:
- `modal deploy` workflow works as documented.
- “workflow per request” model works.
- R2 upload + signed URL flow works in real runs.
- Memory snapshot helper is installed and detected.
- Quota and cancellation APIs are functioning.

Partially aligned / needs stronger production posture:
- Multi-tenant auth/ownership model is not production-grade yet.
- SSRF and webhook authenticity are not hardened to enterprise level.
- `.env.example` can induce expensive default ops.

## Production readiness verdict

Current status:
- Functionally stable for controlled usage: **YES**.
- Safe for multi-tenant public SaaS at 2k+ DAU without additional hardening: **NO (not yet)**.

Main blockers to call it "production-ready":
1. Tenant ownership enforcement.
2. Stronger SSRF controls.
3. Stronger cancellation semantics.
4. Signed webhook delivery.
5. Safe/default infra knobs in `.env.example`.

## Artifacts generated in this validation
- `tests_outputs/concurrency3_validation_20260220_015730.json`
- `tests_outputs/concurrency3_run1_9d471b95_0.png`
- `tests_outputs/concurrency3_run2_96266a28_0.png`
- `tests_outputs/concurrency3_run3_79934ba4_0.png`

## Recommended next step (minimal, high impact)
Implement only these compact changes first (no over-refactor):
1. Enforce `user_id` ownership on all job read/list/delete routes.
2. Harden URL fetch validation with DNS+redirect private-range blocking.
3. Add signed webhook header (HMAC).
4. Update `.env.example` to conservative defaults matching intended low-cost operation.

After these 4 changes, re-run the exact same test battery and re-check container scale-down.
