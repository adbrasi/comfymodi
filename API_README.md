# ComfyUI Modal API – Operator Guide

This document describes the production ComfyUI API deployed by
`modal_comfyui_with_sageattention.py`. It covers request formats, progress
tracking, parallel execution, media handling, and operational tips so you can
reuse or extend this stack in future projects.

> **Base URL**
>
> ```text
> https://cezarsaint--comfyui-saas-api-api.modal.run
> ```
>
> All examples below use this host. Replace it with your deployment’s domain as
> needed.

---

## Architecture Overview

| Component                    | Responsibility                                                                 |
| --------------------------- | ------------------------------------------------------------------------------ |
| `ComfyService` (GPU class)  | Boots ComfyUI on an L40S GPU, applies workflows, streams progress, collects outputs. |
| FastAPI app (`api()` func)  | Receives REST requests, persists job metadata on the shared `/jobs` volume, spawns workers. |
| Shared volumes              | `/cache` for models, `/jobs` for job metadata and webhook logs.                 |
| Autoscaling settings        | `max_containers=4`, each container processes one job at a time.                 |

All job requests are persisted immediately. Workers pick up jobs asynchronously
(using `.spawn()`), so API latency remains low even under load.

---

## Endpoints

| Method | Path                      | Description                                                                 |
| ------ | ------------------------- | --------------------------------------------------------------------------- |
| POST   | `/v1/jobs`                | Submit a workflow execution job.                                            |
| GET    | `/v1/jobs/{job_id}`       | Retrieve status, progress, outputs.                                         |
| DELETE | `/v1/jobs/{job_id}`       | Cancel a queued or running job.                                             |
| GET    | `/v1/jobs`                | List recent jobs (optional status filter).                                  |
| GET    | `/health`                 | Lightweight health probe (ComfyUI reachability check).                      |

> **Rate limits**: SlowAPI enforces `10/min` for POST `/v1/jobs`, `120/min` for
> GET `/v1/jobs/{job_id}`, and `30/min` for DELETE/GET `/v1/jobs`. Exceeding the
> limit returns HTTP 429 with `Retry-After` header.

---

## 1. Submitting Jobs (`POST /v1/jobs`)

### Request body schema

```jsonc
{
  "workflow": { /* required – ComfyUI workflow JSON */ },
  "inputs": [  // optional overrides applied before execution
    {
      "node": "16",            // string or int; converted to string internally
      "field": "positive_prompt",
      "value": "Looping animation, vibrant colors",
      "type": "raw"             // "raw" (default) | "image_base64" | "image_url"
    }
  ],
  "media": [   // optional; <= 10 items, <= 50MB each
    {
      "name": "input.png",     // becomes the filename seen by LoadImage nodes
      "data": "<base64>"       // or provide "url": "https://..."
    }
  ],
  "webhook_url": "https://example.com/hook", // optional – see Webhooks section
  "priority": 0                               // reserved for custom queues
}
```

### Media handling rules

| Field          | Behaviour                                                                               |
| -------------- | --------------------------------------------------------------------------------------- |
| `data`         | Base64 decoded into `/root/comfy/ComfyUI/input/<name>`.                                  |
| `url`          | Downloaded with retries, stored under the same path. Content-type must be non-text.      |
| `type=image_*` | Helper for dynamic overrides that writes the image to the per-job input directory.      |
| File size      | Requests above 50 MB are rejected with HTTP 400.                                         |

Workflows exported from ComfyUI (“Save (API Format)”) typically reference local
filenames. Ensure the names in `media` match the expected inputs (e.g. node 107
in the Wan22 workflow expects `107.png`).

### Response

```json
{
  "job_id": "58e0c0c8-b8c2-413f-a3ac-9dc5ba940571",
  "status": "queued",
  "created_at": "2025-09-19T23:17:12.801234+00:00",
  "estimated_time": 30
}
```

The job is immediately persisted to `/jobs/<job_id>.json` and scheduled. A
GPU container picks it up shortly afterwards (cold start ~8–12 s).

---

## 2. Tracking Progress (`GET /v1/jobs/{job_id}`)

The status payload exposes multiple progress signals:

```json
{
  "job_id": "58e0c0c8-b8c2-413f-a3ac-9dc5ba940571",
  "status": "running",
  "progress": 42,                      // cumulative percent (monotonic)
  "progress_value": 9,                 // current node step value
  "progress_max": 24,                  // current node step max
  "nodes_total": 25,
  "nodes_done": 11,
  "current_node": "71",
  "prompt_id": "c0852d3d-8eca-4bd5-ae4d-1f1ba77c7ab0",
  "last_event_time": "2025-09-19T23:18:08.452101+00:00",
  ...
}
```

### How it works

- `progress` is computed as `(executed_nodes + current_partial) / total_nodes`.
  It never decreases and hits 100 when the job is done.
- `progress_value` / `progress_max` mirror ComfyUI’s native WebSocket
  `progress` events (e.g. sampler step counters). Use them for a secondary
  “frame” or “iteration” progress bar if desired.
- `nodes_done` and `nodes_total` give discrete coverage. When all nodes finish,
  `nodes_done == nodes_total`.
- `current_node` identifies the node currently running (`null` once complete).

### Polling cadence

- The worker flushes status to the job volume whenever progress changes or at
  least every 5 seconds. Polling every 2–5 seconds provides smooth updates.
- The included `test_wan22_api.py` script demonstrates polling and renders both
  the cumulative percentage and the live counter:

  ```text
  📊 Progress: [██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  58% | Elapsed: 94.0s | ETA: 67s
  📈 Progress counter: 14/24
  🧮 Nodes: 15/25
  🔧 Current node: 90
  ```

### Real-time webhook option

For server-to-server notifications, specify `webhook_url` at submission time.
The worker posts `job.completed` or `job.failed` events with timestamps. Webhook
payloads do **not** include outputs; follow up with `GET /v1/jobs/{job_id}`.

---

## 3. Retrieving Outputs

When the job completes, the status response includes a list of output artefacts:

```json
"outputs": [
  {
    "filename": "58e0c0c8_videoFinal_00029.mp4",
    "type": "video",
    "size_bytes": 23022124,
    "data": "<base64>"
  },
  {
    "node_id": "69",
    "type": "text",
    "filename": "text_output_69.txt",
    "size_bytes": 112,
    "data": "..."
  }
]
```

To persist a file locally:

```python
import base64, pathlib

def save_output(item, job_id):
    payload = base64.b64decode(item["data"])
    out_path = pathlib.Path("outputs") / f"{job_id}_{item['filename']}"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_bytes(payload)
    return out_path
```

Large videos are streamed from ComfyUI using chunked requests to avoid loading
hundreds of MB into memory. For extremely large artefacts, consider replacing
base64 embedding with a signed download URL that points directly to ComfyUI’s
`/view` endpoint.

---

## 4. Cancelling Jobs (`DELETE /v1/jobs/{job_id}`)

Cancels queued or running jobs:

- Calls ComfyUI’s `/interrupt` and `/queue/cancel` endpoints.
- Updates the job record with `status="failed"` and `error="Cancelled by user"`.
- Sends a webhook (if configured) with `event="job.failed"`.

Completed or failed jobs return HTTP 200 with a message indicating they were
already finalised.

---

## 5. Listing Jobs (`GET /v1/jobs`)

Supports optional filters:

```
GET /v1/jobs?status=running&limit=20
```

Returns the most recent jobs in reverse chronological order. Each item includes
`job_id`, `status`, `created_at`, and `progress`.

---

## 6. Parallel Execution & Scaling

- `max_containers=4` allows four GPU workers in parallel. Increase this value in
  `@app.cls` if you need more concurrency.
- Each container processes one job at a time (`@modal.concurrent(max_inputs=1)`).
  This avoids GPU contention for heavy workflows like Wan22.
- Submit jobs back-to-back to utilise all available containers. The provided
  `batch_submit.py` snippet in the original README (see repository history) is a
  good starting point; adjust `ThreadPoolExecutor` size to your desired level of
  concurrency.
- Monitor Modal usage; each active container incurs GPU billing for the time it
  runs plus a small idle window (`scaledown_window=100`).

---

## 7. Example Clients

### cURL submission

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d @payload.json \
  https://cezarsaint--comfyui-saas-api-api.modal.run/v1/jobs
```

Where `payload.json` contains the JSON body described earlier. The response’s
`job_id` can then be polled with cURL or any HTTP client.

### Minimal Python helper

```python
import base64
import json
import time
import requests

API = "https://cezarsaint--comfyui-saas-api-api.modal.run"

payload = {
    "workflow": json.load(open("workflows/wan22Smoothloop_fixed.json")),
    "media": [
        {
            "name": "107.png",
            "data": base64.b64encode(open("image.png", "rb").read()).decode()
        }
    ]
}

job = requests.post(f"{API}/v1/jobs", json=payload, timeout=30).json()
job_id = job["job_id"]
print("Job:", job_id)

while True:
    status = requests.get(f"{API}/v1/jobs/{job_id}", timeout=10).json()
    print(status["progress"], "%", status["current_node"])
    if status["status"] in {"completed", "failed"}:
        break
    time.sleep(3)
```

### Realtime dashboards

Because progress is polled over HTTP, it is easy to surface in dashboards or
ChatOps bots. The fields to watch are `status`, `progress`, `progress_value`,
`progress_max`, and `current_node`. `last_event_time` helps detect stalled jobs.

---

## 8. Using Different Workflows

1. **Export from ComfyUI**:
   - Enable *Dev Mode Options* in ComfyUI.
   - Use **File → Save (API Format)** to generate JSON.
2. **Reference inputs**: Identify the `LoadImage` nodes and note their expected
   filenames. Provide matching `media` entries when submitting.
3. **Dynamic overrides**: Add entries to `inputs` to replace prompt strings,
   guidance scales, clip weights, etc. Each entry targets a node and field.
4. **Testing**: Before exposing a new workflow, run it locally with
   `test_wan22_api.py` (or fork it) to confirm the pipeline handles media,
   progress, and outputs as expected.

---

## 9. Webhooks

Configure per job via `webhook_url`. On completion/failure the worker sends:

```json
{
  "event": "job.completed",     // or "job.failed"
  "job_id": "...",
  "timestamp": "2025-09-19T23:19:15.147201+00:00"
}
```

Delivery uses exponential backoff (3 attempts). Consider adding a small relay or
queue if you need guaranteed delivery or payload enrichment.

---

## 10. Operational Tips

- **Authentication**: Add middleware (API keys, OAuth) before exposing publicly.
- **Logging**: Worker tail logs are saved in `job_data["error_log_tail"]` if a job
  fails. Extend this to a dedicated log store if needed.
- **Cleanup**: A nightly cron job (`cleanup_old_jobs`) removes job files older
  than 7 days and stale cache artefacts.
- **Scaling knobs**: Adjust `max_containers`, `scaledown_window`, or the Modal
  autoscaler via `modal shell` when traffic spikes.
- **Model downloads**: The image build installs models via Hugging Face and
  Mega. For faster cold starts, pre-populate the `/cache` volume (e.g., run
  `modal run modal_comfyui_with_sageattention.py::download_assets_and_setup`).

---

## 11. Troubleshooting

| Symptom                                      | Likely cause & resolution                                                                 |
| -------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Progress stalls below 100%                   | Check `/v1/jobs/{id}` → `last_event_time`. If stale >30 s, inspect worker logs in Modal; Comfy may have crashed. |
| Repeated 429 responses                       | SlowAPI rate limit triggered. Back off and honour `Retry-After`. Consider raising limits if appropriate. |
| Upload rejected (HTTP 400)                   | Media file missing `data`/`url`, exceeds 50 MB, or content-type is text. Fix payload.      |
| Worker keeps running after “Prompt executed” | With the new logic, this should stop. If it reappears, ensure the `/history` endpoint returns status; inspect logs. |
| GPU warning in API logs                      | API runs on CPU image now. If you see driver warnings, verify the API function decorator uses `api_image`. |

---

## 12. Test Harness (`test_wan22_api.py`)

The bundled script exercises the API end-to-end:

1. Loads `workflows/wan22Smoothloop_fixed.json`.
2. Injects a prompt override.
3. Base64-encodes an image and submits a job.
4. Polls `/v1/jobs/{job_id}` every ~1.5 seconds, printing:
   - Monotonic `progress` bar.
   - Live `progress_value/progress_max` counter.
   - Nodes done vs total, current node ID.
5. Writes returned artefacts to `outputs/`.

Use it as a template for automated tests or CI smoke checks.

---

## 13. Extending the API

- **Authentication**: Add API key headers via FastAPI dependencies and store
  secrets with `modal.Secret`.
- **Custom queues**: Introduce a priority queue or FIFO cutoff by inspecting the
  `/jobs` directory and reordering job dispatch.
- **Streaming previews**: Wrap the WebSocket connection to forward `progress`
  frames via Server-Sent Events or WebSocket proxies if you need push updates.
- **Output storage**: Swap base64 for cloud storage (e.g., upload outputs to S3
  and return signed URLs) to reduce payload size.

With this guide, you can reproduce the deployment, onboard new workflows, or
build richer orchestration around the existing API quickly and safely.


test_wan22_api.py:#!/usr/bin/env python3
"""
Simple test script for the Wan22 Smooth Loop workflow.
"""

import requests
import json
import base64
import time
from pathlib import Path
from typing import Optional

# Your Modal API endpoint
API_URL = "https://cezarsaint--comfyui-saas-api-api.modal.run"

def run_workflow(image_path: str, prompt: str, *, poll_interval: float = 2.0, timeout_seconds: int = 540, debug: bool = False):
    """Submit a job and stream progress to stdout."""

    # Load workflow (with UTF-8 encoding for Windows compatibility)
    workflow_path = Path("workflows") / "wan22Smoothloop_fixed.json"
    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    image_filename = Path(image_path).name

    # Update workflow to use our image
    workflow["107"]["inputs"]["image"] = image_filename

    # Submit job
    print(f"🚀 Submitting job with image: {image_filename}")
    try:
        response = requests.post(
            f"{API_URL}/v1/jobs",
            json={
                "workflow": workflow,
                "inputs": [
                    {
                        "node": "16",
                        "field": "positive_prompt",
                        "value": prompt,
                        "type": "raw"
                    }
                ],
                "media": [
                    {
                        "name": image_filename,
                        "data": image_base64
                    }
                ]
            },
            timeout=30
        )
    except requests.RequestException as e:
        print(f"❌ Error submitting job: {e}")
        return None

    if response.status_code == 429:
        print("❌ Rate limited when submitting job. Please wait and retry.")
        print(f"Response: {response.text}")
        return None

    if response.status_code != 200:
        print(f"❌ Error submitting job: {response.status_code}")
        print(f"Response: {response.text}")
        return None

    try:
        job = response.json()
        job_id = job['job_id']
        print(f"✅ Job ID: {job_id}")
        print(f"📊 Status URL: {API_URL}/v1/jobs/{job_id}")
        print("-" * 60)
    except Exception as e:
        print(f"❌ Failed to parse response: {e}")
        print(f"Response text: {response.text}")
        return None

    # Poll for status
    print("⏳ Job queued, waiting for processing...")
    last_progress = -1
    last_status = None
    last_nodes_done = -1
    last_current_node: Optional[str] = None
    job_prompt_id: Optional[str] = None
    last_progress_value = -1
    last_progress_max = -1
    start_time = time.time()
    poll_count = 0

    while True:
        try:
            response = requests.get(f"{API_URL}/v1/jobs/{job_id}", timeout=10)
        except requests.exceptions.Timeout:
            print(f"\n⚠️ Request timeout, retrying...")
            time.sleep(poll_interval)
            continue
        except requests.RequestException as e:
            print(f"\n⚠️ Error getting status: {e}")
            time.sleep(poll_interval)
            continue

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else poll_interval * 2
            print(f"\n⏳ Rate limited, retrying in {delay:.1f}s...")
            time.sleep(delay)
            continue

        if response.status_code >= 500:
            print(f"\n⚠️ Server error {response.status_code}, retrying...")
            time.sleep(poll_interval * 1.5)
            continue

        if response.status_code != 200:
            print(f"\n⚠️ Failed to get status: {response.status_code}")
            print(f"Response: {response.text}")
            time.sleep(poll_interval)
            continue

        try:
            status = response.json()
        except ValueError as e:
            print(f"\n⚠️ Failed to parse status JSON: {e}")
            time.sleep(poll_interval)
            continue

        poll_count += 1

        if job_prompt_id is None and status.get('prompt_id'):
            job_prompt_id = status['prompt_id']
            print(f"🆔 Prompt ID: {job_prompt_id}")

        nodes_total = status.get('nodes_total') or 0
        nodes_done = status.get('nodes_done') or 0
        current_node = status.get('current_node')
        progress_value = status.get('progress_value') or 0
        progress_max = status.get('progress_max') or 0

        if status['status'] == 'completed':
            # Clear the progress line if we were showing it
            if last_status == 'running':
                print()  # New line after progress bar
            
            print(f"✨ Completed!")
            print(f"⏱️ Total time: {time.time() - start_time:.1f} seconds")
            print(f"📊 Total API calls: {poll_count}")
            if job_prompt_id:
                print(f"🆔 Prompt ID: {job_prompt_id}")
            if nodes_total:
                print(f"🧮 Nodes completed: {nodes_done}/{nodes_total}")
            if progress_max:
                print(f"📈 Progress values: {progress_value}/{progress_max}")

            # Create outputs directory
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)

            # Save outputs (videos are base64 encoded in the response)
            saved_files = []
            for idx, output in enumerate(status.get('outputs', [])):
                if output.get('data'):
                    try:
                        # Determine file extension from type or filename
                        filename = output.get('filename', f'output_{idx}.mp4')
                        file_type = output.get('type', 'video')

                        # Ensure .mp4 extension for videos
                        if file_type == 'video' and not filename.endswith(('.mp4', '.webm', '.avi')):
                            filename = f"{filename}.mp4"

                        output_path = output_dir / f"{job_id}_{filename}"

                        # Decode base64 video data and save
                        data = output['data']

                        # Handle potential padding issues
                        try:
                            video_bytes = base64.b64decode(data)
                        except:
                            # Try adding padding if needed
                            missing_padding = len(data) % 4
                            if missing_padding:
                                data += '=' * (4 - missing_padding)
                            video_bytes = base64.b64decode(data)

                        with open(output_path, 'wb') as f:
                            f.write(video_bytes)

                        size_mb = len(video_bytes) / (1024 * 1024)
                        print(f"💾 Saved {file_type}: {output_path} ({size_mb:.2f} MB)")
                        saved_files.append(str(output_path))
                    except Exception as e:
                        print(f"⚠️ Could not save output {idx}: {e}")
                        continue

            if saved_files:
                print(f"\n✅ Successfully saved {len(saved_files)} file(s)")
            return status

        elif status['status'] == 'failed':
            print(f"\n❌ Failed: {status.get('error')}")
            print(f"⏱️ Failed after: {time.time() - start_time:.1f} seconds")
            tail = status.get('error_log_tail') or []
            if tail:
                print("🔍 Error log tail (most recent lines):")
                for line in tail[-10:]:
                    print(f"   {line}")
            return status

        # Show progress
        progress = status.get('progress', 0)
        if progress_max:
            computed = int((progress_value / progress_max) * 100)
            progress = max(progress, computed)
        current_status = status['status']

        # Show status changes
        if current_status != last_status:
            if current_status == 'running':
                print("🚀 Job started processing!")
                print("-" * 60)
            elif current_status == 'queued':
                # Don't spam queued messages
                if poll_count == 1 or poll_count % 10 == 0:
                    print(f"\r⏳ Job still queued... (poll #{poll_count})", end='', flush=True)
            last_status = current_status

        # Show progress updates for running jobs
        if current_status == 'running':
            if nodes_total and nodes_done != last_nodes_done:
                print(f"\n🧮 Nodes: {nodes_done}/{nodes_total}")
                last_nodes_done = nodes_done

            if current_node != last_current_node:
                if current_node:
                    print(f"\n🔧 Current node: {current_node}")
                last_current_node = current_node

            if progress_max and (
                progress_value != last_progress_value or progress_max != last_progress_max
            ):
                print(f"\n📈 Progress counter: {progress_value}/{progress_max}")
                last_progress_value = progress_value
                last_progress_max = progress_max

            # Always show progress updates if it changed
            if progress != last_progress:
                # Create progress bar
                bar_length = 40
                filled = int(bar_length * progress / 100) if progress > 0 else 0
                bar = '█' * filled + '░' * (bar_length - filled)
                
                # Calculate ETA
                elapsed = time.time() - start_time
                if progress > 0:
                    eta = (elapsed / progress) * (100 - progress)
                    eta_str = f" | ETA: {eta:.0f}s"
                else:
                    eta_str = ""
                
                msg = f"\r📊 Progress: [{bar}] {progress:>3}% | Elapsed: {elapsed:.1f}s{eta_str}"
                print(msg, end='', flush=True)
                last_progress = progress
            
            # Debug mode shows more details
            if debug and poll_count % 5 == 0:
                print(f"\n[DEBUG] Poll #{poll_count}: Status={current_status}, Progress={progress}%", end='')

        if time.time() - start_time > timeout_seconds:
            print(f"\n⏱️ Timeout after {timeout_seconds} seconds!")
            print(f"📊 Made {poll_count} API calls")
            return None

        time.sleep(poll_interval)

if __name__ == "__main__":
    # Example usage
    # Make sure you have an image file!

    # Create a simple test image if none exists
    test_image = r"C:\Users\dodo-\Downloads\2loras_test__00163_.png"
    if not Path(test_image).exists():
        print("Creating a test image...")
        try:
            from PIL import Image
            img = Image.new('RGB', (512, 512), color='blue')
            img.save(test_image)
            print(f"✅ Created test image: {test_image}")
        except ImportError:
            print("⚠️ PIL not installed. Install with: pip install Pillow")
            print("Please provide your own image file as 'test_image.jpg'")
            exit(1)

    # Run the workflow (set debug=True to see detailed status)
    result = run_workflow(
        image_path=test_image,
        prompt="IntenseAnimation,a dynamic and smooth animation with a lot of movement, looping. colorful, vibrant.",
        debug=True,  # Set to True for debugging
        poll_interval=1.5  # Faster polling for better progress updates
    )

    if result:
        print("\n🎉 Done! Check the output files.")
