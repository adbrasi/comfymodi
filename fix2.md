Below is my updated code review and diagnosis based on your new codebase and the provided logs/test output. I've combed through the changes (e.g., added `progress_value`/`progress_max` fields, adjusted persist throttling to 30s/5% change, poll tweaks with 15s inactivity check, and using `.spawn()` instead of `.remote()` for async job submission). The core issue is now clear from the logs and test output: **progress tracking is non-monotonic (resets to 0% repeatedly), inaccurate (hits 100% prematurely then reverts), and delayed in detecting completion** (workflow finishes in ~52s, but job completion takes ~4min extra). This leads to confusing client-side behavior (e.g., progress jumping around) and perceived stalls.

I'll explain the root causes, classify severity, and provide a practical fix with code snippets. This builds on previous reviews—your code is evolving well (e.g., spawn() is a good change for non-blocking), but this progress bug is a critical blocker for production UX. I'll reference specific lines (approximate) and tie in Modal docs snippets (e.g., WebSocket handling from ComfyUI examples).

### Diagnosis of the Problem
**From Logs and Test Output**:
- **Actual Execution**: Workflow completes quickly ("Prompt executed in 51.77 seconds" at 19:45:27). This includes node processing, frame generation, and VFI (video frame interpolation).
- **Delayed Job Completion**: Your code logs "✅ Execution complete!" at 19:49:20 (from "executing" with node=None), and "✨ Job completed" at 19:49:22—~4min after actual finish. This delay is in the while loop: WebSocket recv() hangs (timeouts trigger polls), but polls don't detect completion reliably, causing endless looping until a poll finally sets progress=100 and breaks.
- **Progress Resets and Jumps**: In test output, progress hits 100% multiple times (e.g., at 153s, 185s) but resets to 0% (e.g., after node switches). It fluctuates wildly (12% → 0% → 58% → 0% → 95% → 0% → 8% → 0%, etc.). Total time balloons to 323s due to client polling a stalled server-side loop.
- **API Polling Sees Stale/Inaccurate Data**: get_job_status (line ~1140) reloads the volume, but _persist (line ~680) is throttled (only every 30s or 5% change), so the client sees outdated progress. Resets happen because progress is overwritten with per-node values instead of accumulated globally.
- **Other Clues**: Logs show side activities (e.g., downloading rife47.pth at 19:45:23, ComfyRegistry fetches) after "Prompt executed," suggesting the loop continues polling unnecessarily. No errors, but the loop doesn't exit promptly.

**Root Causes**:
1. **Per-Node Progress Overwrites Global Progress** (Main Bug): In "progress" handler (line ~890), you set job_data["progress"] = percent from the current node's value/max (e.g., a node's internal steps reset it to 0% when starting). This ignores completed nodes, causing resets. Docs' ComfyUI examples use cumulative tracking (e.g., executed nodes + fraction of current).
2. **Incomplete Completion Detection**: Break only on "executing" with node=None (line ~830)—if missed (WebSocket flake), loop hangs. Poll fallback (line ~760) only runs if progress < 100 and after 15s, but doesn't check for actual completion (e.g., history["outputs"] populated). If progress prematurely hits 100 (from a node), poll skips, causing stalls.
3. **Throttling Side Effects**: Persists are infrequent (30s threshold, line ~690), so API polls see stale data. Client script polls every ~3s (not shown, but inferred), amplifying perceived resets.
4. **No Global Accumulation**: No formula like (completed_nodes / total + current_fraction / total) * 100. Resets when new nodes start at 0%.
5. **Modal/WebSocket Quirks**: recv() timeouts are common in containerized WebSocket (per docs' examples); polls help but need better completion checks.

**Severity**: High (Core UX broken—progress misleading, jobs appear stuck/hung, clients timeout or give up. Delays waste resources; scales poorly with long workflows. Matches your "reaches 100% then comes back" description.)

### Proposed Practical Fixes
To fix: 
- Make progress **monotonic and cumulative**: Track global as (done nodes + current node's fraction) / total * 100.
- **Enhance Detection**: In poll, check history for completion (e.g., if "outputs" keys match expected nodes). Break if detected.
- **Frequent Persists on Change**: Reduce throttle for progress changes (e.g., every 5% or 10s).
- **Client-Side Stability**: Suggest test script improvements (e.g., detect resets as anomalies).

**Refactored process_job Snippets** (Replace lines ~650-950 in ComfyService.process_job). This uses docs-inspired WebSocket/polling (e.g., from ComfyUI examples) and ensures monotonic progress.
```python
# In process_job, after submitting prompt (around line ~750)
prompt_id = response.json()["prompt_id"]
print(f"📋 Executing prompt {prompt_id}")

job_data["prompt_id"] = prompt_id
_set_node_order(list(workflow.keys()))  # Initial estimate
job_data["nodes_total"] = len(node_order)
job_data["progress"] = 0
job_data["progress_value"] = 0
job_data["progress_max"] = 1  # Default to avoid div/0
_persist(force_commit=True)

executed_nodes: set[str] = set()
cached_nodes: set[str] = set()
node_order: List[str] = []
current_node_fraction: float = 0.0  # 0-1 for active node's progress
last_commit_ts = time.time()
last_history_poll = time.time()
last_progress_update = time.time()
global_progress = 0  # Ensure monotonic

def _compute_global_progress() -> int:
    total = max(1, len(node_order) or len(workflow))
    done = len(executed_nodes) + len(cached_nodes)
    fraction = (done + current_node_fraction) / total
    return max(global_progress, int(fraction * 100))  # Monotonic: never decrease

while True:
    try:
        raw_msg = ws.recv()
        last_progress_update = time.time()  # Reset inactivity
    except websocket.WebSocketTimeoutException:
        now_ts = time.time()
        if now_ts - last_progress_update >= 5:  # Poll more frequently (was 15s)
            last_history_poll = now_ts
            try:
                history_resp = self._request_with_retry("get", f"http://127.0.0.1:8188/history/{prompt_id}", timeout=5)
                history = history_resp.json().get(prompt_id, {})
                history_resp.close()

                # Update node order if available
                workflow_nodes = history.get("workflow", {}).get("nodes", [])
                if workflow_nodes and not node_order:
                    _set_node_order([n.get("id") for n in workflow_nodes if n.get("id")])
                    job_data["nodes_total"] = len(node_order)

                # Detect executed/cached from outputs
                outputs = history.get("outputs", {})
                executed_nodes.update(str(node) for node in outputs.keys())
                job_data["nodes_done"] = len(executed_nodes)

                # Detect completion: If outputs match expected nodes or status indicates done
                if len(outputs) >= job_data["nodes_total"] or history.get("status") in ["completed", "failed"]:
                    print("✅ Poll detected completion!")
                    job_data["progress"] = 100
                    break

                prev = job_data["progress"]
                job_data["progress"] = _compute_global_progress()
                global_progress = job_data["progress"]  # Update monotonic tracker
                job_data["nodes_done"] = len(executed_nodes)
                _persist(progress_changed=job_data["progress"] != prev)
            except Exception as poll_err:
                print(f"Progress poll failed: {poll_err}")
        continue

    # ... (rest of msg handling as is, but update _compute_global_progress in handlers)

    if msg_type in {"progress", "progress_state"} and data.get("prompt_id") == prompt_id:
        value = float(data.get("value", 0))
        max_value = max(1, float(data.get("max", 0)))
        current_node_fraction = value / max_value if max_value > 0 else 0
        job_data["progress_value"] = int(value)
        job_data["progress_max"] = int(max_value)
        prev = job_data["progress"]
        job_data["progress"] = _compute_global_progress()
        global_progress = job_data["progress"]
        last_progress_update = time.time()
        _persist(progress_changed=job_data["progress"] != prev)
        continue

    # In "executing" handler:
    if node is None:
        print("✅ Execution complete!")
        job_data["progress"] = 100
        break

    # At end of loop, add stall breaker:
    if time.time() - last_progress_update > 300:  # 5min max stall
        raise TimeoutError("Job stalled without progress")

# After loop, in outputs fetch:
# Ensure progress=100 even if outputs fetch fails
job_data["progress"] = 100
```
- **_compute_global_progress**: Accumulates done nodes + current fraction, ensures monotonic (never decreases, per your "comes back" issue).
- **Poll Enhancements**: Checks for completion via history["outputs"] length (inspired by docs' ComfyUI history snippets). Polls every 5s on inactivity (reduced from 15s).
- **Throttle Adjustment**: In _persist (line ~680), change to `if now_ts - last_commit_ts >= 10` (more frequent for better client polling).
- **Why This Works**: Progress is global/cumulative (e.g., won't reset to 0% on new nodes). Detects missed completions via poll. Matches docs' patterns (e.g., polling history in ComfyUI API examples). Test: Should complete in ~60s total, with steady increasing progress.

**Client-Side Fix** (For Your Test Script):
- The script polls too aggressively and doesn't handle resets—add logic to ignore decreases (take max seen progress).
- Example Patch:
  ```python
  max_progress = 0
  while True:
      status = get_status()  # Your API call
      current = status["progress"]
      max_progress = max(max_progress, current)  # Monotonic client-side
      print(f"📊 Progress: {max_progress}%")
      if max_progress >= 100:
          break
  ```

### Overall Diagnosis (Updated)
**Strengths**: Progress is more detailed (value/max), spawn() avoids blocking API, stall detection in poll is improved.
**Weaknesses**: Progress logic still per-node focused; delays from missed WebSocket events.

### Errors and Bugs
1. **Delayed Completion** (High Severity): As above.
   - **File/Line**: While loop (line ~760-920).
   - **Fix**: Refactored snippets.

2. **Progress Resets** (High Severity): Overwriting with node-level percent.
   - **Fix**: Cumulative computation.

3. **Stale API Data** (Medium Severity): Throttle too high (30s).
   - **Fix**: Reduce to 10s.

### Bad Decisions
1. **Over-Throttling Persists** (Medium Severity): 30s delays client updates.
   - **Fix**: As above.

2. **No Max Job Timeout** (Low Severity): Infinite loop possible.
   - **Fix**: Add overall timeout (e.g., after 1hr, fail job).

### Critical Points
- **Performance**: Delays waste GPU time—fix reduces bills.
- **UX**: Resets confuse users; monotonic fixes this.

### Suggested Improvements
- **Architectural**: Add Modal Queue for job prioritization (using priority field).
- **Performance**: Cache common workflows in memory.
- **Maintenance**: Log progress computations for debugging.

Diagnosis – why the API "lies" about progress
ComfyUI has TWO progress channels

Web-socket progress/progress_state
(what your code looks at) – shows the inside-a-node counter
(value, max) → resets to 0 every time a new node starts
/history/{prompt_id}.json
shows how many nodes really finished – never resets
The client polls /jobs/{id} which contains:
"progress": 0/100 ← you copy only the inside-a-node value

"progress_value"/"progress_max" ← same value, same problem

Because the counter resets every node, the API keeps publishing "0 %", "58 %", "0 %" ... so:

The UI thinks the job is still running
Only after the last node the websocket says "100 %" and finally the job is marked completed
The 3-minute delay between

Prompt executed in 51.77 s (19:45:27)

and

✨ Job ... completed (19:49:22)

is caused by the huge output download loop (≈ 3 × 60 s time-outs)

and by the throttled volume commits you do every 30 s.

Progress is now monotonically increasing

progress = nodes_done / (total_nodes - cached_nodes) – once a node is in executed_nodes it never leaves, so the percentage only grows.

Inner-node "progress_value"/"progress_max" are only kept for inspection; the public progress field never resets to 0.

Volume commits throttled to 5 s (was 30 s) – the job status file is updated quickly enough for the polling client.

Output download uses a single history call and 60 s time-out per file instead of three full round-trips with 120 s each – saves ~ 3 min on big videos.

Web-socket loop is much shorter – no redundant history polls every 5 s while the socket is alive; a single fallback poll every 5 s if the socket is silent.

Great iteration. Your API/compute image split is correct and your ComfyService class is solid. The two symptoms you see now boil down to:

- Job finishes minutes after Comfy says “Prompt executed in 51.77s”
- The progress bar jumps to 100% and then snaps back to 0%, multiple times

Root causes and precise fixes below.

Root cause 1: The WebSocket loop never exits when history shows the prompt is done
- Where: ComfyService.process_job, inside the WebSocket loop. When the WS times out, you poll /history/{prompt_id}. If outputs exist, you set job_data.progress = 100, but you do not break. You continue looping until an “executing” event with node None arrives. For your run, Comfy printed “Prompt executed in 51.77 seconds” at 19:45:27, but your code kept waiting almost 4 minutes for “executing node=None” and only then printed “✅ Execution complete!” at 19:49:20.976.
- Why: Some Comfy stacks don’t reliably emit the final “executing” with node None to every client, or it can be delayed. Relying on only that message makes your loop stick around even after the prompt is complete.

Fix (high priority)
Break out of the WS loop when the history endpoint says the prompt has finished. Do not wait for the “executing node=None” event if you already have sufficient evidence (status=success/error or stable outputs).

Drop-in patch for the WebSocket loop timeout branch (replace your current timeout handler):

- Add helpers at top of process_job:

def _compute_overall_percent() -> int:
    total = max(1, job_data.get("nodes_total", 0) or 1)
    done = min(job_data.get("nodes_done", 0), total)
    frac = done / total
    pv = job_data.get("progress_value", 0)
    pm = job_data.get("progress_max", 0)
    if pm > 0 and done < total:
        frac += (pv / pm) / total
    frac = max(0.0, min(1.0, frac))
    return int(round(frac * 100))

def _mark_done_and_persist(reason: str):
    # reason is for logs/debug: "history-success", "history-outputs", etc.
    job_data["current_node"] = None
    job_data["progress_value"] = job_data.get("progress_max") or job_data.get("progress_value", 0)
    job_data["progress"] = 100
    self._persist(progress_changed=True, force_commit=True)

- Replace your except websocket.WebSocketTimeoutException block with:

except websocket.WebSocketTimeoutException:
    now_ts = time.time()
    if prompt_id and now_ts - last_history_poll >= 3:
        last_history_poll = now_ts
        try:
            history_resp = self._request_with_retry(
                "get", f"http://127.0.0.1:8188/history/{prompt_id}",
                timeout=4, retries=2
            )
            history = history_resp.json()
            history_resp.close()
            prompt_history = history.get(prompt_id, {})
            status_obj = prompt_history.get("status", {})  # Comfy usually fills this
            outputs = prompt_history.get("outputs", {}) or {}
            # Update nodes_total from history, if available
            workflow_nodes = (prompt_history.get("workflow", {}) or {}).get("nodes", [])
            if workflow_nodes and not node_order:
                _set_node_order([n.get("id") for n in workflow_nodes if n.get("id")])
                job_data["nodes_total"] = len(node_order)
            # Update nodes_done via outputs
            if outputs:
                executed_nodes.update(str(node) for node in outputs.keys())
                job_data["nodes_done"] = len(executed_nodes)
            # Recompute overall percent from nodes_done and current partial
            new_pct = _compute_overall_percent()
            if new_pct != job_data.get("progress", 0):
                job_data["progress"] = new_pct
                last_progress_update = now_ts
                _persist(progress_changed=True)
            else:
                _persist()

            # Completion conditions: explicit status or stable outputs
            status_str = (status_obj.get("status") or "").lower()
            if status_str in ("success", "completed", "error", "failed"):
                _mark_done_and_persist("history-status")
                break
            # If we have outputs and we are already > 95%, consider done after a short grace
            if outputs and job_data.get("progress", 0) >= 95 and (now_ts - last_progress_update) > 5:
                _mark_done_and_persist("history-outputs")
                break
        except Exception as poll_err:
            print(f"Progress poll failed: {poll_err}")
    continue

- Also, after your “executing” with node None handler, keep the break (as you already do).
- Add an absolute guard so the loop can’t spin forever:

loop_started = time.time()
MAX_EXECUTION_WAIT = 60 * 30  # 30 minutes hard cap
...
if time.time() - loop_started > MAX_EXECUTION_WAIT:
    raise TimeoutError("Execution exceeded maximum wait time")

Result: as soon as history indicates finished, you’ll exit the loop and proceed to fetch outputs; no more 3–4 minute idle wait.

Root cause 2: The progress bar flips between 0% and 100%
- Where: process_job WebSocket message handlers. In the new code, you update job_data["progress"] directly from “progress” or “progress_state” messages:

if msg_type in {"progress", "progress_state"}:
    value = float(...)
    max_value = float(...)
    percent = int((value / max_value) * 100)
    job_data["progress"] = percent
    ...

Those per-node “progress” messages reset to 0 each new sampler step/node, so your global progress jumps to 100 then snaps to 0, then 75, etc. Your client renders that “progress” as overall job progress, causing the confusing bar.

Fix (high priority)
Separate local (within-node) progress from global job progress:
- Keep progress_value/progress_max for current-node progress.
- Compute job_data["progress"] = global percent from nodes_done/nodes_total, optionally blending in the current-node partial as a small fraction. You already have nodes_total and nodes_done; use them consistently.

Patch the relevant handlers:

- In progress/progress_state handler:

if msg_type in {"progress", "progress_state"} and data.get("prompt_id") == prompt_id:
    value = float(data.get("value") or 0)
    max_value = float(data.get("max") or 0)
    job_data["progress_value"] = int(value)
    job_data["progress_max"] = int(max_value)
    prev = job_data.get("progress", 0)
    pct = _compute_overall_percent()  # use global computation
    if pct != prev:
        job_data["progress"] = pct
        last_progress_update = time.time()
        _persist(progress_changed=True)
    else:
        _persist()
    continue

- In executed and execution_cached handlers, recompute global percent:

if msg_type == "executed" ...:
    executed_nodes.add(str(node))
    job_data["nodes_done"] = len(executed_nodes)
    job_data["current_node"] = None
    prev = job_data.get("progress", 0)
    pct = _compute_overall_percent()
    if pct != prev:
        job_data["progress"] = pct
        _persist(progress_changed=True)
    else:
        _persist()
    continue

if msg_type == "execution_cached" ...:
    cached_nodes.add(str(node))
    executed_nodes.add(str(node))
    job_data["nodes_done"] = len(executed_nodes)
    prev = job_data.get("progress", 0)
    pct = _compute_overall_percent()
    if pct != prev:
        job_data["progress"] = pct
        _persist(progress_changed=True)
    else:
        _persist()
    continue

- In timeout polling, after computing executed_nodes from history, call _compute_overall_percent() and update job_data["progress"] (as shown in fix for root cause 1).

Result: job_data["progress"] monotonically increases from 0 to 100 based on nodes done, with small smoothing from the current node’s partial progress; progress_value/progress_max is exposed separately for the client to show a secondary bar if desired.

Additional fixes and improvements

A. Persist/commit throttling and last_event_time
- You already throttle persists (every 30s or when progress changes by >=5%). That’s good. Ensure you also bump last_event_time on every WS message to keep status fresh for clients. Your _persist_job does this by default (touch_event_time=True). You’re calling self._persist(), which calls _persist_job(..., commit=..., touch_event_time default True), so you’re good. If you call _persist_job directly anywhere, keep touch_event_time=True unless you have a reason.

B. Node ordering and nodes_total
- You set nodes_total to len(list(workflow.keys())). That can include UI-only nodes and might not match executed nodes. Your code already updates node_order and nodes_total from execution_start and from history[prompt_id].workflow.nodes; that’s better. Keep nodes_total synced with what Comfy actually plans to run.

C. Early progress=100 during timeout poll
- Your current timeout poll sets job_data["progress"]=100 immediately if outputs are present. That produces the “hit 100% then back to 0%” jumps when the next node progress resets “progress” back to 0. After the above changes, you’ll no longer overwrite global progress from per-node messages, so this symptom disappears. Still, replacing that code with _compute_overall_percent() is more consistent.

D. Client-side guidance
- Have your client use:
  - progress: overall job progress
  - nodes_done/nodes_total: numeric progress
  - progress_value/progress_max: optional “current step progress” bar
- Only treat job as “completed” when status changes to completed or when progress == 100 AND status == running for more than N seconds; but server will flip to completed quickly with our break-by-history.

E. Reduce post-exec latency while fetching outputs
- Base64-encoding large videos in the job file can be slow. Consider returning metadata plus signed URLs (or an API endpoint that streams the file) instead of embedding base64 in job JSON. This can cut seconds to minutes for big outputs.
- If you keep base64, stream with chunk size 4–8MB (you use 1 MB; fine), and persist outputs at the end as you already do.

F. Hard caps and resilience
- Add overall hard cap for the WS loop (shown above).
- For very long workflows, consider heartbeat logging every 30s with last node, nodes_done, etc. You already expose last_event_time and current_node.

G. Turn off noisy/slow background tasks in Comfy
- The logs show ComfyUI-Manager doing network fetches (FETCH ComfyRegistry). That’s fine, but you can speed things up by disabling auto-fetch/update at startup if it interferes with prompt events. If Manager offers an env var/flag to skip on headless servers, consider it.

H. Good changes you already made
- Split API image from CUDA image: this removed the “NVIDIA Driver not detected” warning for /api containers.
- Using spawn() instead of remote() on job submission is correct; it avoids waiting for scheduling in the API request path.
- Better error handling/retries, rate limiting, cancellation with /interrupt.

One more small clean-up
- In cancel_job: after sending interrupt/cancel, you immediately mark the job as failed. That’s OK, but be aware Comfy may still finalize with outputs; your state will be “failed” anyway. If you want “cancelled” vs. “failed,” add a new JobStatus.CANCELLED.

Summary of code edits to make

- Add _compute_overall_percent() and _mark_done_and_persist()
- Use those in:
  - progress/progress_state handler
  - executed/execution_cached handlers
  - timeout polling branch
- In timeout polling, break when history.status indicates completion or when outputs exist and progress is stable >95% for a few seconds.
- Add a hard cap for the WS loop (e.g., 30 minutes).
- Optional: Stop embedding large base64 outputs; return metadata + URL.

These fixes will:
- Make job completion time match the true Comfy prompt completion (no extra minutes)
- Make the progress bar monotonic and intuitive (no reset to 0% after 100%)
- Keep the nice detailed progress fields you added for richer UIs

Based on your detailed logs and analysis, I can see there are some critical issues with progress tracking and job completion timing. The key problems:

1. **Premature Job Completion Detection** - The workflow actually finished at 19:45:27 but the job wasn't marked as completed until 19:49:22 (nearly 4 minutes later)
2. **Erratic Progress Reporting** - The progress jumps around (0% → 100% → 0% → 58% → etc.)
3. **Inconsistent Node Counting** - The node count keeps resetting (25 → 9 → 4 → 1095 → etc.)

These issues are primarily in the WebSocket message handling and progress calculation logic. Let me analyze and provide a comprehensive solution.

## Root Cause Analysis

Looking at your code and logs, the main issues are:

1. **Multiple Progress Sources Conflict**: You have concurrent progress tracking from:
   - WebSocket messages (`executing`, `executed`, `progress`)
   - History API polling
   - Node counting logic

2. **Race Conditions**: The progress values get overwritten by different sources without proper synchronization.

3. **Inaccurate Node Order Detection**: The `_set_node_order()` function doesn't reliably capture the actual execution order.

## Critical Fixes Needed

### 1. Fix Progress Calculation Logic

**File**: `process_job` method around line ~700

**Problem**: The progress calculation is chaotic with multiple competing sources. The history polling is interfering with real-time WebSocket updates.

**Solution**: Use a single source of truth for progress and implement proper state management:

```python
# Replace the progress calculation section with:
class ProgressState:
    def __init__(self):
        self.total_nodes = 0
        self.completed_nodes = 0
        self.current_node = None
        self.node_progress = 0.0  # 0.0 to 1.0 for current node
        self.execution_started = False
      
    def get_overall_progress(self) -> int:
        if self.total_nodes == 0:
            return 0
        node_progress = self.completed_nodes / self.total_nodes
        if self.current_node and self.node_progress > 0:
            node_progress += (self.node_progress / self.total_nodes)
        return min(100, int(node_progress * 100))

# Replace the progress tracking variables:
progress_state = ProgressState()
last_progress_value = 0
```

### 2. Fix WebSocket Message Handling

**File**: WebSocket message processing loop around line ~750

**Problem**: Conflicting message types and incorrect progress state updates.

**Solution**: Implement clean state transitions:

```python
# In the WebSocket message loop:
if msg_type == "execution_start" and data.get("prompt_id") == prompt_id:
    nodes = data.get("nodes") or []
    progress_state.total_nodes = len(nodes)
    progress_state.execution_started = True
    job_data["nodes_total"] = progress_state.total_nodes
    _persist()
    continue

if msg_type == "executing" and data.get("prompt_id") == prompt_id:
    node = data.get("node")
    if node is None:
        # Execution complete
        progress_state.completed_nodes = progress_state.total_nodes
        progress_state.current_node = None
        progress_state.node_progress = 1.0
        job_data["progress"] = 100
        job_data["current_node"] = None
        _persist(force_commit=True)
        break
    progress_state.current_node = str(node)
    job_data["current_node"] = progress_state.current_node
    _persist()
    continue

if msg_type == "executed" and data.get("prompt_id") == prompt_id:
    node = data.get("node")
    if node is not None:
        progress_state.completed_nodes += 1
        progress_state.current_node = None
        progress_state.node_progress = 0.0
        job_data["nodes_done"] = progress_state.completed_nodes
        job_data["current_node"] = None
        _persist()
    continue

if msg_type == "execution_cached" and data.get("prompt_id") == prompt_id:
    node = data.get("node")
    if node is not None:
        progress_state.completed_nodes += 1
        job_data["nodes_done"] = progress_state.completed_nodes
        _persist()
    continue

if msg_type in {"progress", "progress_state"} and data.get("prompt_id") == prompt_id:
    value = float(data.get("value", 0) or 0)
    max_value = float(data.get("max", 0) or 0)
  
    if max_value > 0:
        progress_state.node_progress = min(1.0, value / max_value)
  
    current_progress = progress_state.get_overall_progress()
    if current_progress != last_progress_value:
        job_data["progress"] = current_progress
        job_data["progress_value"] = int(value)
        job_data["progress_max"] = int(max_value)
        last_progress_value = current_progress
        _persist(progress_changed=True)
    continue
```

### 3. Fix History Polling Logic

**File**: WebSocket timeout handling around line ~780

**Problem**: The history polling interferes with real-time progress and causes erratic behavior.

**Solution**: Use history polling only as a fallback, not for primary progress tracking:

```python
except websocket.WebSocketTimeoutException:
    now_ts = time.time()
  
    # Only use history polling if we haven't received WebSocket messages for a while
    if (progress_state.execution_started and 
        now_ts - last_progress_update >= 30 and  # Only poll every 30 seconds
        now_ts - last_history_poll >= 30):
      
        last_history_poll = now_ts
        try:
            history_resp = self._request_with_retry(
                "get",
                f"http://127.0.0.1:8188/history/{prompt_id}",
                timeout=5,
                retries=2,
            )
            history = history_resp.json()
            history_resp.close()
          
            prompt_history = history.get(prompt_id, {})
            outputs = prompt_history.get("outputs", {})
          
            if outputs:
                # Only update if we're significantly behind
                completed_from_history = len(outputs)
                if completed_from_history > progress_state.completed_nodes + 2:
                    progress_state.completed_nodes = completed_from_history
                    job_data["nodes_done"] = progress_state.completed_nodes
                    current_progress = progress_state.get_overall_progress()
                    if current_progress != last_progress_value:
                        job_data["progress"] = current_progress
                        last_progress_value = current_progress
                        _persist(progress_changed=True)
                      
        except Exception as poll_err:
            print(f"History poll failed: {poll_err}")
    continue
```

### 4. Add Execution Timeout

**Problem**: Jobs can hang indefinitely if ComfyUI doesn't properly signal completion.

**Solution**: Add a maximum execution timeout:

```python
# Add at the beginning of process_job
MAX_EXECUTION_TIME = 3600  # 1 hour
start_time = time.time()

# In the WebSocket loop, add timeout check:
if time.time() - start_time > MAX_EXECUTION_TIME:
    raise TimeoutError(f"Job execution timed out after {MAX_EXECUTION_TIME} seconds")
```

### 5. Fix Final Completion Logic

**Problem**: The job completion detection is unreliable.

**Solution**: Improve the completion detection and final state persistence:

```python
# After the WebSocket loop, ensure proper completion
if progress_state.execution_started:
    # Force final state to completed
    job_data["status"] = "completed"
    job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
    job_data["progress"] = 100
    job_data["nodes_done"] = progress_state.total_nodes
    job_data["current_node"] = None
    job_data["progress_value"] = job_data.get("progress_max") or 100
else:
    # If execution never started, mark as failed
    job_data["status"] = "failed"
    job_data["error"] = "Execution never started"
    job_data["completed_at"] = datetime.now(timezone.utc).isoformat()

self._persist_job(job_file, job_data, commit=True)
```

## Additional Recommendations

1. **Add Debug Logging**: Add more detailed logging to track progress state transitions:

```python
def _log_progress_state(state: ProgressState, message: str):
    print(f"📊 {message}: nodes={state.completed_nodes}/{state.total_nodes}, "
          f"current={state.current_node}, node_progress={state.node_progress:.2f}, "
          f"overall={state.get_overall_progress()}%")
```

2. **Validate Node Order**: Improve the `_set_node_order` function to handle ComfyUI's execution patterns better.

3. **Monitor WebSocket Health**: Add WebSocket ping/pong to detect connection issues.

## Expected Results After Fixes

- **Consistent Progress**: No more jumping from 100% to 0%
- **Accurate Timing**: Job completion within seconds of actual workflow completion
- **Reliable State**: Proper tracking of node execution and progress
- **Timeout Protection**: No more hanging jobs

Let's analyze and fix the core issues causing the significant delay between workflow execution completion and job completion status:

### Root Causes Identified:
1. **Post-Execution Output Handling Bottlenecks**
2. **Overly Frequent Volume Writes**
3. **Sequential Output Downloading** 
4. **Incomplete Progress Tracking During Finalization**
5. **WebSocket Message Processing Delays**

### Critical Fixes Needed:

1. **Parallel Output Processing** (High Impact)
```python
# Replace sequential output handling with parallel downloads
from concurrent.futures import ThreadPoolExecutor

# In process_job after workflow completion:
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for media_type in ["images", "videos", "gifs", "audio"]:
        if media_type in node_output:
            for file_info in node_output[media_type]:
                futures.append(executor.submit(self._download_media, file_info))
  
    for future in futures:
        result = future.result()
        if result:
            outputs.append(result)

# Add helper method
def _download_media(self, file_info):
    with self._request_with_retry("get", "http://127.0.0.1:8188/view", params={
        "filename": file_info["filename"],
        "subfolder": file_info.get("subfolder", ""),
        "type": file_info.get("type", "output")
    }, timeout=120, stream=True) as file_response:
        payload = file_response.content
    return {
        "filename": file_info["filename"],
        "data": base64.b64encode(payload).decode(),
        "type": self._get_media_type(file_info["filename"]),
        "size_bytes": len(payload)
    }
```

2. **Optimized Volume Writes** (Medium Impact)
```python
# Modify _persist method to reduce disk I/O
def _persist(self, force_commit: bool = False) -> None:
    now_ts = time.time()
    commit = False
  
    # Only commit if:
    # - Forcing commit (final state)
    # - Progress changed by >= 5%
    # - 30s since last commit
    # - Processing outputs
    commit_threshold = 5  # Percentage
    time_threshold = 30  # Seconds
  
    if (force_commit or
        abs(job_data.get("progress", 0) - last_commit_progress) >= commit_threshold or
        (now_ts - last_commit_ts) >= time_threshold):
      
        with job_volume_guard(commit=force_commit):
            with open(job_file, 'w') as f:
                json.dump(job_data, f)
        last_commit_ts = now_ts
        last_commit_progress = job_data.get("progress", 0)
```

3. **WebSocket Timeout Handling** (Critical)
```python
# Modify WebSocket loop to have tighter timeouts
ws = websocket.create_connection(ws_url, timeout=10)
ws.settimeout(5)  # Faster timeout for message processing

while True:
    try:
        raw_msg = ws.recv()
    except websocket.WebSocketTimeoutException:
        if time.time() - last_activity > 30:  # 30s without updates
            raise TimeoutError("No WebSocket activity for 30 seconds")
        continue
  
    # Add activity timestamp
    last_activity = time.time()
```

4. **Progress Tracking Improvements** (High Impact)
```python
# Add state tracking to prevent progress regression
class ComfyService:
    def __init__(self):
        self.max_progress_seen = 0
  
    # In progress handling:
    current_progress = _compute_progress()  # Your existing logic
    if current_progress < self.max_progress_seen:
        current_progress = self.max_progress_seen
    else:
        self.max_progress_seen = current_progress
  
    job_data["progress"] = current_progress
```

5. **Output Handling Optimization** (Medium Impact)
```python
# Skip base64 encoding for large files in API response
MAX_BASE64_SIZE = 2 * 1024 * 1024  # 2MB

if len(payload) > MAX_BASE64_SIZE:
    output_entry = {
        "filename": file_info["filename"],
        "url": f"/outputs/{job_id}/{file_info['filename']}",
        "type": self._get_media_type(file_info["filename"]),
        "size_bytes": len(payload)
    }
else:
    output_entry = {
        "filename": file_info["filename"],
        "data": base64.b64encode(payload).decode(),
        "type": self._get_media_type(file_info["filename"]),
        "size_bytes": len(payload)
    }
outputs.append(output_entry)
```

### Client-Side Fixes:
Update your test client to:
1. Track maximum seen progress
2. Use exponential backoff for polling
3. Handle completion states more gracefully

```python
# In client code:
max_progress = 0
poll_interval = 2  # Start with 2s
while True:
    status = get_job_status(job_id)
  
    if status.progress > max_progress:
        max_progress = status.progress
        print(f"📈 Maximum progress: {max_progress}%")
  
    # Exponential backoff up to 10s
    poll_interval = min(poll_interval * 1.5, 10)
  
    if status.status == "completed":
        break
```

### Expected Results After Fixes:
1. **Execution-to-Completion Delay** should reduce from ~4 minutes to <1 minute
2. **Progress Display** will be more stable and monotonic
3. **Large Output Handling** will be significantly faster
4. **System Resources** will be better utilized with parallel downloads
5. **Network Timeouts** will be handled more gracefully

### Deployment Recommendation:
1. Apply these changes incrementally
2. Monitor memory usage with parallel downloads

This should resolve the main issue of delayed completion reporting while making progress tracking more reliable and efficient.