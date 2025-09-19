CODES:
modal_comfyui_with_sageattention.py = Main modal code
test_wan22_api.py = Code to test

You will read this file and create a todo list to implement all the recommended improvements. Some improvements are duplicated because I sent the same code to several friends, and each has a different opinion. You will read and understand everything, and then create a plan to fix our modal_comfyui_with_sageattention.py and then fix test_wan22_api.py for us to use next.



### Key Response to Your Main Issue: Progress Tracking
You proposed counting total nodes in the workflow and incrementing progress (+1 per executed node) until 100%. This is a good idea—it's simple and aligns with ComfyUI's "executing" WebSocket messages, which fire per node. Your current implementation in `process_job` (around lines 600-650) already attempts this:
```python
nodes_total = len(workflow)
nodes_completed = 0
# In WebSocket loop:
if msg_type == "executing":
    # ...
    nodes_completed += 1
    progress = int((nodes_completed / nodes_total) * 100)
    # Update job file
```
However, you report it "never works" (e.g., no progress visible in JS/Py code). Based on debugging similar ComfyUI setups (and Modal docs snippets like those for WebSocket handling), potential causes:
- **WebSocket Message Issues**: ComfyUI's WebSocket might not send "executing" for every node reliably, especially with custom nodes or long-running workflows. It skips binary messages, but some workflows (e.g., with previews) might flood with unhandled types.
- **Concurrency/Race Conditions**: Multiple threads (via `@modal.concurrent`) could interfere with WebSocket recv() or file writes to the shared job volume.
- **Modal-Specific Quirks**: Snapshots or GPU restores might disrupt the WebSocket connection.
- **Client-Side Visibility**: Your JS code (not provided) might not poll `/v1/jobs/{job_id}` frequently enough, or the API doesn't reload the volume in real-time.

**Proposed Fix**: Enhance the WebSocket loop for robustness and real-time progress. Use a more accurate node tracking method by parsing ComfyUI's history API periodically (as a fallback). Here's a refactored snippet for `process_job` (replace lines ~600-700). This uses the docs' WebSocket patterns and adds polling for reliability:
```python
# In process_job, after submitting prompt
nodes_total = len(workflow)
nodes_completed = 0
last_progress_update = time.time()

while True:
    try:
        # Primary: WebSocket for real-time updates
        raw_msg = ws.recv(timeout=1)  # Add timeout to allow polling
        if isinstance(raw_msg, bytes):
            continue
        msg = json.loads(raw_msg)
        if msg.get("type") == "executing" and msg["data"].get("prompt_id") == prompt_id:
            node = msg["data"].get("node")
            if node is None:
                break  # Done
            nodes_completed += 1
            progress = int((nodes_completed / nodes_total) * 100)
            print(f"🔧 Node {node} executed ({nodes_completed}/{nodes_total}) - {progress}%")
            # Update job file (throttle to every 5s to reduce volume commits)
            if time.time() - last_progress_update > 5:
                job_data["progress"] = progress
                with job_volume_guard(commit=True):
                    with open(job_file, 'w') as f:
                        json.dump(job_data, f)
                last_progress_update = time.time()

    except websocket.WebSocketTimeoutException:
        # Fallback: Poll history API every 5s if WebSocket stalls
        if time.time() - last_progress_update > 5:
            try:
                history = requests.get(f"http://localhost:8188/history/{prompt_id}", timeout=2).json()
                executed_nodes = len(history.get(prompt_id, {}).get("outputs", {}))  # More accurate count
                progress = int((executed_nodes / nodes_total) * 100)
                job_data["progress"] = progress
                with job_volume_guard(commit=True):
                    with open(job_file, 'w') as f:
                        json.dump(job_data, f)
                last_progress_update = time.time()
            except Exception as e:
                print(f"Progress poll failed: {e}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        break
```
- **Why This Works**: Combines real-time WebSocket with periodic polling (inspired by Modal's ComfyUI examples in docs). Throttles file writes to avoid overload. For JS client, poll `/v1/jobs/{job_id}` every 2-5s.
- **Test**: Run a simple workflow and monitor logs. If still failing, add `ws.settimeout(10)` and debug with `websocket.enableTrace(True)`.
- **Performance Impact**: Minimal; polling is lightweight.

For the GPU warning ("WARNING: The NVIDIA Driver was not detected."), this often occurs if the image lacks proper CUDA setup or if snapshots restore without GPU. See fix in "Errors and Bugs" below.

Now, the full review.

### Overall Diagnosis
**Strengths**:
- Excellent use of Modal features: Volumes for persistence, snapshots for speed, concurrent for parallelism, background tasks for async job processing.
- Robust job lifecycle: Queued -> Running -> Completed/Failed with webhooks.
- Good separation of concerns: API in FastAPI, core logic in `ComfyService` cls.
- Handles media uploads and dynamic inputs well.

**Weaknesses**:
- Build-time model downloads are slow and error-prone (timeouts, Mega.nz unreliability).
- Concurrency issues with shared resources (e.g., job volume, WebSocket).
- Progress tracking is brittle (as you noted).
- Error handling is inconsistent (e.g., silent webhook failures).
- Performance: No caching for repeated workflows; high cold-start times without snapshots fully utilized.

Estimated Improvements: With fixes, execution time could drop 20-50% (better caching/snapshots), errors reduced by 70% (better handling), and production-readiness high.

### Errors and Bugs
1. **GPU Detection Warning** (High Severity): "WARNING: The NVIDIA Driver was not detected." This appears in snapshots or enter() methods without GPU (e.g., `setup_environment` runs `snap=True` without GPU). From docs (e.g., Flux example), GPU is only available post-snapshot.
   - **File/Line**: `setup_environment` (line ~400) and `start_comfy_with_gpu` (line ~450).
   - **Fix**: Move GPU-dependent imports (e.g., torch) to `@modal.enter(snap=False)`. Add `gpu="L40S"` to the cls decorator if not already (it is, but ensure it's propagated). Test with `modal run verify_setup`.

2. **WebSocket Reliability** (Medium Severity): recv() can hang or miss messages, leading to stuck jobs or inaccurate progress (your main issue).
   - **File/Line**: `process_job` WebSocket loop (line ~600).
   - **Fix**: See progress tracking proposal above. Add timeouts and retries (e.g., `ws = websocket.create_connection(..., timeout=30)`).


### Critical Points
- **Snapshot Reliability**: `enable_memory_snapshot=True` is great (10x speed per docs), but GPU init in `start_comfy_with_gpu` could fail post-restore. Test with long-running jobs.


### Suggested Improvements
**Architectural**:

- **Progress API**: Enhance `/v1/jobs/{job_id}` to stream progress via WebSocket (docs have examples).

**Performance**:
- **Cold Start**: Your snapshots are good; add `warm_pool_size=2` to cls for faster scaling.
- **Execution Time**: Parallelize media downloads with asyncio (docs have aiohttp examples).

**Maintenance**:
- **Logging**: Use structlog or Modal's logging for better traceability.

ComfyUI progress is never visible to the caller
“Minor” code changes break everything
Cold-start & run-time efficiency can still be improved

CRITICAL – Progress Is Lost in a Black Box
Root cause

ComfyService.process_job() only counts how many WebSocket “executing” messages arrive.

That number is not the true node count; ComfyUI sends one “executing” per prompt-id, not per node.

Therefore progress always jumps 0 → 100 % and the front-end receives no intermediate value.

File:

comfyui-saas-api.py:566-575

Fix (≤ 15 lines, 0 external deps)

a) Send the workflow to /prompt with the extra flag "extra_data": {"extra_pnginfo": {"workflow": workflow}} – this forces ComfyUI to echo the exact node list.

b) After the prompt call, parse /history/{prompt_id} once and keep the node_order array.

c) In the WebSocket loop map every data["node"] to its index in that array → progress = 100 * (idx+1) / len(node_order).

Patch snippet (drop-in replacement block inside process_job):

<PYTHON>
# ---------- 1. submit ----------
r = requests.post("http://localhost:8188/prompt",
                  json={"prompt": workflow, "client_id": client_id,
                        "extra_data": {"extra_pnginfo": {"workflow": workflow}}},
                  timeout=30)
prompt_id = r.json()["prompt_id"]
# ---------- 2. get ordered node list ----------
history = requests.get(f"http://localhost:8188/history/{prompt_id}").json()
node_order = list(history[prompt_id]["outputs"])          # execution order
total = len(node_order)
# ---------- 3. progress loop ----------
while True:
    msg = json.loads(ws.recv())
    if msg.get("type") == "executing" and msg["data"]["prompt_id"] == prompt_id:
        node_id = msg["data"]["node"]
        if node_id is None:            # finished
            break
        progress = int((node_order.index(node_id) + 1) * 100 / total)
        job_data["progress"] = progress
        with job_volume_guard(commit=True):
            (job_file).write_text(json.dumps(job_data))
HIGH – Race Between Container Memory-Snapshot & GPU
Root cause

@modal.enter(snap=True) must not touch GPU/CUDA.

The very first import of torch or any CUDA call makes the snapshot uncacheable on GPU workers → every cold start re-runs the full build.

You already import torch inside setup_environment().

File:

comfyui-saas-api.py:220

Fix

Move every GPU import into the non-snapshot enter method:

<PYTHON>
@modal.enter(snap=True)
def setup_environment(self):
    import sys, os, pathlib                     # <- CPU only
    os.environ["COMFYUI_PATH"] = "/root/comfy/ComfyUI"
    ...
    #  NO torch, NO transformers here
@modal.enter(snap=False)                       # GPU available
def start_comfy_with_gpu(self):
    import torch                               # <- first CUDA touch
    ...
HIGH – Fragile ComfyUI Start-Up
Root cause

Sub-process is spawned but never reaped; if ComfyUI aborts the container keeps answering health-checks and the user sees an infinite “running 0 %” job.

File:

comfyui-saas-api.py:250-270

Fix

Use subprocess.Popen as context manager and add a liveness probe:

<PYTHON>
with subprocess.Popen(["python", "/root/comfy/ComfyUI/main.py",
                       "--listen", "0.0.0.0", "--port", "8188"],
                      stdout=subprocess.PIPE,
                      stderr=subprocess.STDOUT) as proc:
    self.process = proc
    # ---- liveness thread ----
    def _watch():
        for line in iter(proc.stdout.readline, b""):
            print(line.decode(), end="")
            if b"Traceback" in line or proc.poll() is not None:
                requests.post("http://localhost:8188/interrupt")   # try clean stop
                modal.experimental.stop_fetching_inputs()
    threading.Thread(target=_watch, daemon=True).start()
    ...
MEDIUM – Redundant Volume Commit Traffic
Root cause

You commit() on every 1 % progress tick.

Modal volumes are eventually consistent; over-committing only adds latency.

Fix

Commit once when status changes or every 10 % step:

<PYTHON>
if progress % 10 == 0 or status in {"completed", "failed"}:
    with job_volume_guard(commit=True):
        job_file.write_text(json.dumps(job_data))
else:
    with job_volume_guard(commit=False):   # only write, no commit
        job_file.write_text(json.dumps(job_data))


Progress tracking logic is wrong/incomplete. You’re incrementing on "executing" messages and never using "executed"/"execution_start"/"execution_cached"/"progress" events, and your WebSocket has no timeout/keepalive. Result: progress often stays at 0, stalls, or "never works".
Mutable defaults in Pydantic models. You define default [] on request models, which can leak state between requests.
Snapshot stage imports torch. This triggers the “NVIDIA Driver was not detected” warning because during snap the GPU isn’t present. Not fatal but noisy and confusing.
Cancel endpoint doesn’t cancel anything. You’re marking a job as failed in the JSON, but you never tell Comfy to stop executing the prompt.
A few architectural/perf fragilities: single input directory, brittle custom-node installs, short timeouts, excessive volume commits, no throttling of job file writes, no prompt_id stored for later status/cancel.
Critical issues and fixes

Progress never visible or fragile (severity: critical) Cause:
You count "executing" events as “node completed”. "executing" fires at node start, not completion.
You don’t use the "execution_start" event (which provides the planned node list), nor "execution_cached" (skip cached nodes), nor "executed" (fires at completion), nor "progress" (per-node granular progress).
WebSocket not configured with timeout, ping, or reconnection. You catch websocket.WebSocketTimeoutException, but you never set a timeout, so it won’t trigger.
Fix: Use the ComfyUI WS protocol properly, with a single source of truth for total/cached/executed nodes and per-node progress. Also throttle writes to the job file to avoid racing/IO overhead.

Replace your WS loop in ComfyService.process_job with this more robust version (drop-in replacement for the "Execute via WebSocket" section):

Execute via WebSocket with robust progress tracking
client_id = str(uuid.uuid4())
ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
ws = websocket.create_connection(ws_url, timeout=15)
ws.settimeout(30)

try:

Submit prompt
resp = requests.post(
"http://127.0.0.1:8188/prompt",
json={"prompt": workflow, "client_id": client_id},
timeout=60,
)
resp.raise_for_status()
prompt_id = resp.json()["prompt_id"]

Persist prompt_id to allow cancel and better status
job_data["prompt_id"] = prompt_id
job_data["nodes_total"] = 0
job_data["nodes_done"] = 0
job_data["current_node"] = None
last_write = 0.0

with job_volume_guard(commit=True):
with open(job_file, "w") as f:
json.dump(job_data, f)

Track execution
planned_nodes = set()        # nodes to run from execution_start
cached_nodes = set()         # nodes reported as cached
executed_nodes = set()       # nodes we’ve seen completed
current_partial = (0, 1)     # (value, max) for current node

def compute_progress():
total = max(1, len(planned_nodes) - len(cached_nodes))
done = min(total, len(executed_nodes))

include in-node progress if available
val, mx = current_partial
frac = done / total
if mx and mx > 0 and done < total:
frac += (val / mx) / total
return int(max(0, min(100, round(frac * 100))))

while True:
raw = ws.recv()
if isinstance(raw, bytes):
continue

msg = json.loads(raw)
mtype = msg.get("type")
data = msg.get("data", {})

When run starts, we get the planned execution nodes
if mtype == "execution_start" and data.get("prompt_id") == prompt_id:
nodes = data.get("nodes") or []
planned_nodes = set(nodes)

On the GET /v1/jobs/{job_id} endpoint, also return prompt_id, nodes_total, nodes_done, current_node so frontend can display richer progress.

Mutable defaults in Pydantic (severity: critical) Cause:
You set defaults to [] in JobRequest; these lists are shared between instances.
Fix in the API schema section:

from pydantic import BaseModel, Field

class JobRequest(BaseModel):
workflow: Dict
inputs: List[WorkflowInput] = Field(default_factory=list)
media: List[MediaFile] = Field(default_factory=list)
webhook_url: Optional[str] = None
priority: int = 0

“NVIDIA Driver was not detected” warning (severity: medium; noisy) Cause:
You import torch during the snap=True enter phase when GPU is unavailable. Torch detects no driver and logs a warning.
Fix options (pick one):

Don’t import torch in setup_environment. Pre-import numpy/PIL/opencv only. Move torch import to start_comfy_with_gpu or later.
Or set os.environ["CUDA_VISIBLE_DEVICES"] = "" before importing torch in snap=True, or set TORCH_LOGS=+nothing to silence. I recommend simply removing the torch import from the snapshot phase.
Cancel endpoint does not cancel jobs (severity: high) Cause:
You mark the JSON as failed but the running Comfy prompt continues.
Fix:

Persist prompt_id in the job JSON (see progress fix).
On cancel, if status is running and prompt_id present, call Comfy’s interrupt endpoint before flipping status.
Example in cancel_job:

Read job_data, if job_data["status"] == "running" and job_data.get("prompt_id"): do: requests.post("http://127.0.0.1:8188/interrupt", timeout=5) requests.post("http://127.0.0.1:8188/queue/cancel", json={"prompt_id": job_data["prompt_id"]}, timeout=5)
Then mark status failed/cancelled.
Note: exact endpoints may differ by Comfy version; commonly /interrupt stops current execution and /queue can cancel. If not available, you can POST /prompt with "clear": true to clear queue; check your Comfy build.

High/medium severity issues and improvements
5) Use per-job input subdirectories (severity: high)
Cause:

All uploaded inputs go into COMFY_DIR/input, so parallel containers or jobs could collide or pick up wrong files if names clash.
Fix:

Make an INPUT_DIR/job_id subfolder and reference that in workflow inputs. Cleanup after job.
In process_job, before handling media:

job_input_dir = INPUT_DIR / job_id
job_input_dir.mkdir(parents=True, exist_ok=True)

Then write files to job_input_dir. For dynamic inputs, save to that folder and set workflow fields to relative paths Comfy resolves within the input dir.

Too frequent volume commits and writes (severity: medium) Cause:
You commit the volume on every progress update. This is slow and increases error surfaces.
Fix:

Throttle progress writes (already shown), and keep commit=True but only every 0.5–1s. For final status, commit once. This reduces I/O and races.
Store and return richer status (severity: medium) Add to job_data:
prompt_id, nodes_total, nodes_done, current_node
last_event_time to help frontends detect staleness/ETAs.
Return these in JobStatusResponse. Your frontend can derive ETA using moving average.

Robust server start and health (severity: medium)
Increase startup wait window to 120–180s for the first-time load with many custom nodes. Models/nodes can take longer than 60s.
Use a persistent health check with a requests.Session and exponential backoff.
Start Comfy the “Modal way” for better lifecycle (severity: medium) Your subprocess call to python main.py is fine, but comfy launch --background is more resilient and prepares for memory snapshot usage:
Replace in start_comfy_with_gpu:

cmd = "comfy launch --background -- --listen 0.0.0.0 --port 8188 --use-sage-attention"
self.process = subprocess.Popen(cmd, shell=True)

Then keep your /system_stats loop. This aligns with Modal’s comfy example and makes future migration to memory snapshots (with GPU rebind) easier.

Memory snapshot: do it right or remove (severity: medium)
enable_memory_snapshot=True only helps substantially if you launch Comfy in the snap=True phase and then rebind GPU after restore using /cuda/set_device, and include the memory_snapshot_helper custom node. Right now you avoid starting Comfy during snapshot, so benefit is limited while still showing torch’s driver warning.
Either add the helper, start Comfy in snap=True and call /cuda/set_device in snap=False (see Modal docs), or turn off enable_memory_snapshot and keep your current two-stage start.
Custom node installs are brittle (severity: medium)
60–120s timeouts for git clone/pip install often fail for heavy repos; shallow clone can miss submodules; HEAD changes break builds.
Prefer pinning to commits or using "comfy node install --fast-deps <node>@<version>" where possible. If you keep git clone, specify a commit/tag per repo and remove the hard 60s timeout or increase to 180–300s.
API image duplication (severity: low)
You pip-install fastapi twice (base image and API function). Use the same image for the API function or remove the duplicate pip_install("fastapi[standard]") on the function to avoid rebuild divergence.

Output fetch robustness (severity: low)

Large videos over /view can time out. Use stream=True and chunked reads to avoid holding large responses in memory.
Example:

with requests.get(view_url, params=..., stream=True, timeout=120) as r:
r.raise_for_status()
content = b''.join(r.iter_content(1 << 20))

Better error logs and job output on failure (severity: low)
Capture stderr/stdout from the Comfy server and attach last few lines to job_data["error_log_tail"] on error; extremely useful for debugging.
Validation and size limits (severity: low)
You estimate base64 size and cap at 50MB; good. Add content-type/extension checks for URLs to prevent incorrect file types from breaking nodes.
Concrete code patches to apply
A) Fix Pydantic defaults (in API models)

from pydantic import BaseModel, Field

class JobRequest(BaseModel):
workflow: Dict
inputs: List[WorkflowInput] = Field(default_factory=list)
media: List[MediaFile] = Field(default_factory=list)
webhook_url: Optional[str] = None
priority: int = 0

B) Don’t import torch in snapshot

In setup_environment (snap=True):

Remove “import torch”.

Keep only light CPU libs (numpy, PIL).

If you want to silence the warning but keep torch import, do:

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch

C) Persist prompt_id and richer status
In process_job after receiving prompt_id:

job_data["prompt_id"] = prompt_id
job_data["nodes_total"] = 0
job_data["nodes_done"] = 0
job_data["current_node"] = None

… and include these fields in JobStatusResponse (and FastAPI response model) so UIs can build better progress bars.

D) Real cancel
In cancel_job, if running and prompt_id exists:

if job_data["status"] == "running" and job_data.get("prompt_id"):
try:

stop current work
requests.post("http://127.0.0.1:8188/interrupt", timeout=5)

cancel enqueued prompt
requests.post("http://127.0.0.1:8188/queue/cancel", json={"prompt_id": job_data["prompt_id"]}, timeout=5)
except Exception as e:
print(f"Cancel failed: {e}")

E) Per-job input dir
In process_job:

job_input_dir = INPUT_DIR / job_id
job_input_dir.mkdir(parents=True, exist_ok=True)

Then write all media and dynamic input files into job_input_dir and reference those names in workflow.

F) Increase server start patience
In start_comfy_with_gpu, bump wait budget to 120–180s and add exponential backoff for retries.

G) Throttle job progress writes
Use the throttling included in the WS loop above.

H) Health checks on every job
Add a lightweight health checker at the start of process_job:

try:
requests.get("http://127.0.0.1:8188/system_stats", timeout=3).raise_for_status()
except Exception:
raise RuntimeError("ComfyUI not healthy in this container")

Architecture/performance notes

Consider moving to "comfy run --workflow <file> --wait" for batch-only flows and parse stdout for progress if the WS protocol becomes flaky for you. The WS protocol is richer, though, and with the above loop it becomes reliable.
If cold-starts are still slow, adopt Modal’s memory snapshot helper and rebind GPU after restore as in the docs (you already enabled enable_memory_snapshot). Without running Comfy in the snapshot, the speedups are limited.
Pin versions for custom nodes and models to avoid “any minor change breaks the code”.
Add a small exponential-backoff retry wrapper for requests.get/post in a shared helper to reduce flakiness under load.
Persist a light-weight in-memory cache of workflow templates if you reuse them often, to avoid reprocessing/re-validating each time.
Why the GPU warning shows up

It’s a warning emitted when torch imports in an environment that ships CUDA libraries but currently has no NVIDIA driver (Modal’s snapshot phase runs without GPU). It does not break anything. Avoid importing torch during snapshot to remove the log, or silence as above.
What you get after these changes

Accurate progress: consistent 0–100% using executed nodes, with smooth intra-node updates from "progress" events.
Stable WS: timeouts, throttled writes, and fewer flakey failures.
Cancellations actually stop Comfy.
No mutable-default foot-guns. Fewer weird “minor change breaks everything” failures.
Cleaner logs (no noisy driver warning).
Better multi-request safety (per-job input dirs) and less volume churn.

🚨 CRITICAL ISSUES
1. GPU Detection and Memory Snapshot Problem
Severity: Critical

File: ComfyService class, setup_environment() method

The warning "The NVIDIA Driver was not detected" appears because you're trying to import GPU-dependent libraries (torch, numpy, cv2) during the snapshot phase when GPU isn't available yet.

Fix:

<PYTHON>
@modal.enter(snap=True)
def setup_environment(self):
    """Setup environment WITHOUT GPU-dependent imports"""
    import sys
    import os
    
    print("📸 Creating memory snapshot (no GPU available yet)...")
    
    # Only import lightweight, non-GPU libraries
    import json  # Safe import
    import time   # Safe import
    
    # Set environment variables
    os.environ["COMFYUI_PATH"] = "/root/comfy/ComfyUI"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    print("✅ Environment snapshot created (GPU libraries will load when available)")
    self.process = None
2. Progress Tracking Implementation
Severity: High

File: process_job() method

Your progress tracking approach is fundamentally flawed. ComfyUI's WebSocket messages don't guarantee sequential node execution, and counting nodes assumes linear execution.

Proper Solution:

<PYTHON>
def process_job(self, job_id: str):
    # ... existing code ...
    
    # Get actual total nodes from workflow
    nodes_total = len([n for n in workflow.values() if isinstance(n, dict) and n.get("class_type")])
    
    # Track executed nodes by ID
    executed_nodes = set()
    
    while True:
        try:
            raw_msg = ws.recv()
            if isinstance(raw_msg, bytes):
                continue
            
            msg = json.loads(raw_msg)
            msg_type = msg.get("type")
            
            if msg_type == "executing":
                data = msg.get("data", {})
                if data.get("prompt_id") == prompt_id:
                    node_id = data.get("node")
                    
                    if node_id is None:
                        print("✅ Execution complete!")
                        break
                    
                    # Track unique node executions
                    if node_id not in executed_nodes:
                        executed_nodes.add(node_id)
                        progress = int((len(executed_nodes) / nodes_total) * 100)
                        progress = min(100, progress)  # Ensure max 100%
                        
                        # Update progress
                        job_data["progress"] = progress
                        with job_volume_guard(commit=True):
                            with open(job_file, 'w') as f:
                                json.dump(job_data, f)
        # ... rest of loop ...
3. WebSocket Connection Handling
Severity: High

File: process_job() method

Current WebSocket handling lacks proper error recovery and timeout management.

Enhanced WebSocket Handling:

<PYTHON>
# Replace the WebSocket section with:
max_retries = 3
for attempt in range(max_retries):
    try:
        ws = websocket.WebSocket()
        ws.settimeout(30)  # Set timeout
        ws.connect(f"ws://localhost:8188/ws?clientId={client_id}")
        
        # Submit prompt
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30
        )
        
        # ... rest of execution logic ...
        
        break  # Success, break retry loop
        
    except (websocket.WebSocketException, requests.RequestException) as e:
        if attempt == max_retries - 1:
            raise Exception(f"WebSocket connection failed after {max_retries} attempts: {e}")
        time.sleep(2 ** attempt)  # Exponential backoff


Critical Issues & Improvements:

1. GPU Detection Warning & Initialization

Cause: The NVIDIA Driver warning occurs because CUDA initialization isn't validated after server start. The snapshot phase (without GPU) taints logs.
Fix:
<PYTHON>
def start_comfy_with_gpu(self):
    ...
    # Add CUDA verification after server starts
    try:
        import torch
        assert torch.cuda.is_available()
        print(f"✅ CUDA initialized. Devices: {torch.cuda.device_count()}")
    except Exception as e:
        raise RuntimeError(f"GPU initialization failed: {e}")
    ...
2. Progress Tracking Improvement

Problem: Node counting is unreliable due to workflow complexity.
Solution: Use ComfyUI's native progress events:
<PYTHON>
while True:
    raw_msg = ws.recv()
    msg = json.loads(raw_msg)
    
    if msg_type == "progress":
        data = msg.get("data", {})
        progress = int(data.get("value", 0) * 100)
        # Update progress in file
        job_data["progress"] = progress
        with job_volume_guard(commit=True):
            with open(job_file, 'w') as f:
                json.dump(job_data, f)
3. WebSocket Handling & Error Recovery

Issue: Fragile connection logic with insufficient error handling.
Enhancement:
<PYTHON>
from websocket import WebSocketTimeoutException, WebSocketConnectionClosedException
MAX_RETRIES = 3
RETRY_DELAY = 2
for attempt in range(MAX_RETRIES):
    try:
        ws = websocket.WebSocket()
        ws.connect(f"ws://localhost:8188/ws?clientId={client_id}", timeout=10)
        # Processing loop here
    except (WebSocketConnectionClosedException, ConnectionRefusedError) as e:
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
            continue
        raise
    finally:
        ws.close()
4. Output Handling Optimization

Problem: Repeated filesystem operations during output collection.
Improvement:
<PYTHON>
# After workflow completion
output_dir = COMFY_DIR / "output"
outputs = []
for file in output_dir.glob("*"):
    if file.suffix in [".png", ".mp4", ".json"]:  # Add more formats as needed
        outputs.append({
            "filename": file.name,
            "data": base64.b64encode(file.read_bytes()).decode(),
            "type": self._get_media_type(file.name),
            "size_bytes": file.stat().st_size
        })
5. Volume Operations Optimization

Issue: Frequent commits causing contention
Solution: Batch volume operations using a temporary directory:
<PYTHON>
from tempfile import TemporaryDirectory
def process_job(self, job_id: str):
    with TemporaryDirectory() as tmpdir:
        # Process files in tmpdir
        # Then atomically move to volume
        with job_volume_guard(reload=True, commit=True):
            shutil.move(tmpdir, JOB_DIR)
Architectural Recommendations:

Add Health Checks

<PYTHON>
@web_app.get("/health/deep")
async def deep_health_check():
    try:
        resp = requests.get("http://localhost:8188/system_stats", timeout=2)
        return {"comfy_status": "healthy" if resp.ok else "unhealthy"}
    except Exception as e:
        raise HTTPException(500, detail=str(e))
Implement Rate Limiting

<PYTHON>
from fastapi import Request
from fastapi.middleware import Middleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
limiter = Limiter(key_func=get_remote_address)
web_app.state.limiter = limiter
web_app.add_exceptionHandler(RateLimitExceeded, _rate_limit_exceeded_handler)
@web_app.post("/v1/jobs")
@limiter.limit("10/minute")
async def create_job(request: Request, ...):
    ...
Add Circuit Breakers

<PYTHON>
from circuitbreaker import circuit
@circuit(failure_threshold=3, recovery_timeout=60)
def _send_webhook(url: str, job_id: str, event: str):
    ...
Specific File Modifications:

In start_comfy_with_gpu (app.py)

Add explicit CUDA initialization check
Implement connection backoff strategy
WebSocket Handling (app.py)

Replace node counting with ComfyUI's progress events
Add message version validation
Implement proper reconnect logic
Volume Operations

Use atomic writes with temporary directories
Reduce commit frequency through batch operations
Performance Tips:

Add GPU-Specific Optimizations

<PYTHON>
# In image definition
.run_commands(
    "echo 'Applying CUDA optimizations...'",
    "sed -i 's/\"device\": \"auto\"/\"device\": \"cuda\"/' /root/comfy/ComfyUI/config.yaml"
)
Implement Memory Snapshot Warmup

<PYTHON>
@modal.enter(snap=True)
def warmup(self):
    # Run dummy inference to compile models
    dummy_input = torch.randn(1, 3, 512, 512).to("cuda")
    with torch.no_grad():
        self.model(dummy_input)


        