# ComfyUI Modal API

Production ComfyUI workflows exposed as REST endpoints on Modal. This README reflects the behaviour implemented in `modal_comfyui_with_sageattention.py` and the provided test harness `test_wan22_api.py`.

## Base URL

```
https://cezarsaint--comfyui-saas-api-api.modal.run
```

No authentication is currently enforced. If you plan to expose this API publicly, add your own auth (API keys, signed headers, etc.).

## Quick Start

1. Deploy (or redeploy) the stack:
   ```bash
   modal deploy modal_comfyui_with_sageattention.py
   ```
2. Prepare an input image and the workflow JSON (the default repo ships `workflows/wan22Smoothloop_fixed.json`).
3. Run the smoke test script:
   ```bash
   python3 test_wan22_api.py
   ```
   The script submits the Wan22 Smooth Loop workflow, polls `/v1/jobs/{job_id}`, and saves any generated videos to `outputs/`.

## Job Lifecycle

1. **Submit a job** with the full ComfyUI workflow JSON, optional dynamic inputs, and media assets (base64 or URL).
2. **Receive a `job_id`** immediately. The job is persisted under `/jobs/{job_id}.json` and queued for execution.
3. A worker container picks up the job and launches ComfyUI inside the Modal GPU environment.
4. **Poll `/v1/jobs/{job_id}`** (or listen for webhooks) until the job reaches `completed` or `failed`.
5. When complete, the response includes base64-encoded assets (`images`, `videos`, `audio`, etc.) with metadata so you can save them to disk.

## Endpoint Overview

| Method | Path                 | Description                            |
| ------ | -------------------- | -------------------------------------- |
| POST   | `/v1/jobs`           | Enqueue a ComfyUI workflow execution   |
| GET    | `/v1/jobs/{job_id}`  | Inspect job status and retrieve outputs|
| DELETE | `/v1/jobs/{job_id}`  | Cancel a queued or running job         |
| GET    | `/v1/jobs`           | List the most recent jobs              |
| GET    | `/health`            | Lightweight health probe                |

---

## POST /v1/jobs — Submit a Job

### Request Body

```json
{
  "workflow": { "107": { "inputs": { "image": "input.png" } } },
  "inputs": [
    {
      "node": "16",
      "field": "positive_prompt",
      "value": "Dynamic looping animation",
      "type": "raw"
    }
  ],
  "media": [
    {
      "name": "input.png",
      "data": "<base64-encoded image bytes>"
    }
  ],
  "webhook_url": "https://example.com/webhook/job",
  "priority": 0
}
```

| Field        | Type                 | Notes |
| ------------ | -------------------- | ----- |
| `workflow`   | object (required)    | Full ComfyUI workflow graph. The worker mutates this object in place when dynamic inputs are applied. |
| `inputs`     | array (optional)     | Each entry targets a node field. Supported `type` values:<br>- `raw` (default) — value inserted directly.<br>- `image_base64` — value is base64, saved to `/input`, field updated to the filename.<br>- `image_url` — remote file downloaded, field updated to the filename. |
| `media`      | array (optional)     | Up to 10 items. Provide either `data` (base64) **or** `url`. Files larger than 50 MB are rejected. Filenames must match the Load Image node in your workflow (e.g. node `107` in `wan22Smoothloop_fixed.json`). |
| `webhook_url`| string (optional)    | If present, a POST is sent when the job finishes (payload documented below). |
| `priority`   | integer (optional)   | Persisted with the job. Current worker implementation processes jobs FIFO but reserves this field for future queue tuning. |

### Sample Response

```json
{
  "job_id": "7b3f94f4-d7a6-417b-9b6f-3d760069f789",
  "status": "queued",
  "created_at": "2025-09-19T17:42:00.123456+00:00",
  "estimated_time": 30
}
```

`estimated_time` is a static hint (seconds) returned by the server.

---

## GET /v1/jobs/{job_id} — Job Status & Outputs

### Sample Response (trimmed)

```json
{
  "job_id": "7b3f94f4-d7a6-417b-9b6f-3d760069f789",
  "status": "completed",
  "created_at": "2025-09-19T17:42:00.123456+00:00",
  "started_at": "2025-09-19T17:42:08.502349+00:00",
  "completed_at": "2025-09-19T17:43:15.117921+00:00",
  "progress": 100,
  "outputs": [
    {
      "filename": "Wan22_SmoothLoop_00001.mp4",
      "type": "video",
      "size_bytes": 24839210,
      "data": "<base64-encoded binary>"
    },
    {
      "node_id": "112",
      "type": "text",
      "filename": "text_output_112.txt",
      "data": "..."
    }
  ],
  "error": null
}
```

- While a job is running, `progress` increments based on node execution and sampler progress.
- On failure, `status` becomes `failed` and `error` contains the exception message raised within the worker.
- `outputs` gathers every file returned by the ComfyUI history API (videos, images, gifs, audio, JSON UI payloads, and text). Everything is base64 encoded so you can persist it locally.

---

## Progress Reporting

The worker tracks both node execution and sampler iterations to estimate completion:

- Every job starts with `progress=0`. When ComfyUI reports that a node finished, `nodes_completed` increments and contributes to the overall score.
- When the KSAMPLER (or any component that emits `progress` events) runs, the worker blends node coverage with sampler progress using a 50/50 weight. This mirrors what you see in `log.md`, where messages like `📊 Progress: 37% (Step 6/24)` appear while a sampler step iterates.
- `progress_details` is persisted alongside the job file and exposed through the status endpoint. It includes the current node, number of completed nodes, total nodes, and sampler step. Example:
  ```json
  {
    "progress": 45,
    "progress_details": {
      "current_node": "60",
      "nodes_completed": 17,
      "nodes_total": 25,
      "step": "12/49"
    }
  }
  ```
- Progress can reset or dip slightly when ComfyUI starts a new sampler block or replays cached nodes. Expect bursts of updates rather than a perfectly linear curve.
- The worker writes out job state at most once every 10 seconds, so polling quicker than that may return the same percentage.

If you need tighter estimates, consider enriching the workflow with explicit checkpoints (e.g., lightweight nodes that emit logs) or surfacing elapsed time in your client UI next to the reported percentage.

---

## Batch Submissions & Parallelism

`ComfyService` is declared with `max_containers=4` and `@modal.concurrent(max_inputs=1)`, which means:

- Up to **four** GPU containers can execute jobs in parallel.
- Each container handles one job at a time; additional jobs trigger the autoscaler to start another container (up to four) instead of sharing a runtime.
- Jobs submitted beyond the live slots queue in `/jobs` until capacity frees up.

To run four workflows at once, submit four POST `/v1/jobs` requests in quick succession. The Modal autoscaler launches extra replicas (cold start takes ~8–10 seconds) and spreads the queued jobs across them. A simple Python batch launcher looks like this:

```python
# batch_submit.py
import concurrent.futures
from pathlib import Path

from test_wan22_api import run_workflow  # reuse the tested function

IMAGE = Path("C:/Users/dodo-/Downloads/2loras_test__00163_.png")
PROMPTS = [
    "Looping animation, neon city",
    "Slow pan, watercolor style",
    "Dynamic camera move, soft lighting",
    "Abstract motion, bold colors",
]

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    futures = [
        pool.submit(run_workflow, str(IMAGE), prompt)
        for prompt in PROMPTS
    ]
    for future in futures:
        future.result()
```

Because `run_workflow` already uploads media and polls `/v1/jobs/{job_id}`, this sends four independent jobs and waits for all to finish. Adjust `max_workers` to match the `max_containers` configured in the service.

### Scaling Up Safely

- **More parallel containers:** increase `max_containers` in `modal_comfyui_with_sageattention.py` and redeploy. Modal’s autoscaler honours these decorator values.
- **Per-container concurrency:** the code now keeps `max_inputs=1` so each Wan22 run gets its own container. The `job_volume_guard` lock still protects `/jobs`, so if you experiment with higher `max_inputs` later you will avoid the previous volume error—just monitor GPU memory and ComfyUI stability.
- **Dynamic tweaks:** at runtime you can call `ComfyService().update_autoscaler(max_containers=8)` from a Modal Python shell to temporarily raise the ceiling; the next deploy resets to the decorator values.

Every additional container or higher `max_inputs` multiplies GPU spend roughly linearly. Monitor utilisation and scale down when demand drops.

---

## DELETE /v1/jobs/{job_id} — Cancel a Job

Cancels queued or running jobs by marking them `failed` with `error` set to `"Cancelled by user"`. Completed jobs remain unchanged.

---

## GET /v1/jobs — List Recent Jobs

Query parameters:
- `status` (optional) — filter by `queued`, `running`, `completed`, or `failed`.
- `limit` (optional, default 50) — number of records to return.

Response example:

```json
{
  "jobs": [
    {
      "job_id": "7b3f94f4-d7a6-417b-9b6f-3d760069f789",
      "status": "completed",
      "created_at": "2025-09-19T17:42:00.123456+00:00",
      "progress": 100
    }
  ]
}
```

---

## Webhooks

If `webhook_url` is supplied on job creation, the worker sends a POST request when the job finishes or fails. Payload:

```json
{
  "event": "job.completed",   // or "job.failed"
  "job_id": "7b3f94f4-d7a6-417b-9b6f-3d760069f789",
  "timestamp": "2025-09-19T17:43:15.147201+00:00"
}
```

The webhook does **not** include outputs. Fetch `/v1/jobs/{job_id}` after receiving the event if you need the artifacts or error message.

---

## Handling Outputs

The `test_wan22_api.py` script demonstrates how to:
- Poll the status endpoint until completion or failure.
- Write base64 data to disk, automatically fixing missing padding.
- Name files using the `filename` and `type` supplied by the API.

Adapt that script to integrate with your own storage pipeline (S3, GCS, etc.).

---

## Workflow Tips

- Use the shipped `workflows/wan22Smoothloop_fixed.json` as a template. Node `107` is the Load Image node; set its `inputs.image` to the filename you upload.
- Dynamic inputs allow you to tweak prompts or parameters without duplicating workflows. Ensure node IDs and field names match those in your ComfyUI graph.
- Uploaded media are written to `/root/comfy/ComfyUI/input`. If you reference files elsewhere in the workflow, point to that directory or supply URLs so the worker can download them.
- The worker commits job state to the `job-storage` volume every ~10 seconds, so progress survives restarts.

---

## Scaling & Runtime Configuration

The worker class is declared in `modal_comfyui_with_sageattention.py`:

```python
@app.cls(
    gpu="L40S",
    image=image,
    volumes={"/cache": cache_volume, "/jobs": job_volume},
    enable_memory_snapshot=True,
    scaledown_window=100,
    max_containers=4,
)
@modal.concurrent(max_inputs=1)
class ComfyService:
    ...
```

- Up to four GPU containers run in parallel, one job per container.
- To adjust parallelism or cost, change `max_containers` or the `@modal.concurrent` decorator and redeploy.
- Model assets live in the `comfyui-cache` volume. The build step installs Wan 2.1/2.2 models, Sage Attention, and a curated set of custom nodes.

The API layer itself runs under a separate `@app.function` with its own `max_containers` (default 3) to handle HTTP traffic independently from job workers.

---

## Troubleshooting

- **HTTP 400 on upload**: Check that every media file is ≤ 50 MB once base64 decoding is accounted for, and that your workflow references the same filename.
- **`Model not found` errors**: Confirm the models required by your workflow are present in the cached volume, or extend `download_assets_and_setup()`.
- **Stuck at `queued`**: Ensure there is at least one available GPU container (`max_containers > 0`) and redeploy if needed. Review Modal logs with `modal app logs com-fyui-saas-api`.
- **Volume reload errors (`there are open files preventing the operation`)**: This happens if more than one job runs in the same container. Keep `@modal.concurrent(max_inputs=1)` or add locking around `job_volume.reload()` / `job_volume.commit()` before raising concurrency.
- **Webhook timeouts**: The worker makes a best-effort HTTP POST with a 10-second timeout and no retries—log webhook payloads server-side for resilience.

---

## Related Files

- `modal_comfyui_with_sageattention.py` — Modal app definition, worker logic, and deployment entry points.
- `test_wan22_api.py` — Minimal integration test that exercises the API end to end.
- `workflows/wan22Smoothloop_fixed.json` — Example ComfyUI workflow referenced by both the API and the test script.

Keep this README in sync with code changes; any adjustments to request or response shapes should be documented here.
