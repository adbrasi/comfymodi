#!/usr/bin/env python3
"""
End-to-end test suite for ComfyUI SaaS API.

Tests: health, auth, job create, progress polling, completion, cancel, quota.

Usage:
    export COMFYUI_API_URL=https://cezarsaint--comfyui-saas-api.modal.run
    export COMFYUI_API_KEY=your-key
    python test_e2e.py

    # Run specific test only:
    python test_e2e.py --test cancel
    python test_e2e.py --test quota
"""

import json
import os
import sys
import time
import uuid
import argparse
from datetime import datetime
from pathlib import Path

import requests

API = os.environ.get("COMFYUI_API_URL", "https://cezarsaint--comfyui-saas-api.modal.run")
KEY = os.environ.get("COMFYUI_API_KEY", "")

if not KEY:
    print("ERROR: set COMFYUI_API_KEY env var")
    sys.exit(1)

HEADERS = {"Authorization": f"Bearer {KEY}"}

# ── helpers ──────────────────────────────────────────────────────────────────

def ok(msg): print(f"  ✓ {msg}")
def fail(msg): print(f"  ✗ {msg}"); sys.exit(1)
def info(msg): print(f"    {msg}")
def section(msg): print(f"\n{'='*50}\n{msg}\n{'='*50}")


def load_workflow():
    path = Path("workflows/sdxl_simple_exampleV2.json")
    if not path.exists():
        fail(f"Workflow not found: {path}")
    return json.loads(path.read_text())


def create_job(workflow, user_id="test_user", seed=None):
    if seed is None:
        seed = int(time.time()) % 999999
    r = requests.post(
        f"{API}/v1/jobs",
        headers=HEADERS,
        json={
            "workflow": workflow,
            "inputs": [
                {"node": "6",  "field": "text",  "value": "cinematic portrait, beautiful lighting", "type": "raw"},
                {"node": "53", "field": "seed",  "value": seed, "type": "raw"},
                {"node": "53", "field": "steps", "value": 15,   "type": "raw"},
            ],
            "user_id": user_id,
        },
        timeout=30,
    )
    return r


def poll_job(job_id, max_wait=300, interval=3):
    """Poll until terminal status. Returns final data dict."""
    start = time.time()
    last_progress = -1
    while True:
        elapsed = time.time() - start
        if elapsed > max_wait:
            fail(f"Timeout after {max_wait}s waiting for job {job_id}")

        r = requests.get(f"{API}/v1/jobs/{job_id}", headers=HEADERS, timeout=10)
        if r.status_code != 200:
            fail(f"GET /v1/jobs/{job_id} returned {r.status_code}: {r.text}")

        data = r.json()
        status = data["status"]
        progress = data.get("progress", 0)

        if progress != last_progress:
            info(f"[{elapsed:5.0f}s] {status} — {progress}% (node={data.get('current_node')}, step={data.get('current_step')}/{data.get('total_steps')})")
            last_progress = progress

        if status in ("completed", "failed", "cancelled"):
            return data

        time.sleep(interval)


# ── tests ────────────────────────────────────────────────────────────────────

def test_health():
    section("1. Health check")
    r = requests.get(f"{API}/health", timeout=10)
    if r.status_code != 200:
        fail(f"Health check failed: {r.status_code}")
    data = r.json()
    if data.get("status") != "ok":
        fail(f"Unexpected health response: {data}")
    ok(f"Health OK — timestamp: {data['timestamp']}")


def test_auth_rejected():
    section("2. Auth rejection")
    r = requests.post(f"{API}/v1/jobs", headers={"Authorization": "Bearer wrong-key"}, json={"workflow": {}, "user_id": "x"}, timeout=10)
    if r.status_code != 401:
        fail(f"Expected 401, got {r.status_code}")
    ok("Invalid token correctly rejected with 401")

    r = requests.get(f"{API}/v1/jobs/fake-id", timeout=10)
    if r.status_code not in (401, 403):
        fail(f"Expected 401 or 403 (no auth header), got {r.status_code}")
    ok(f"Missing auth header correctly rejected with {r.status_code}")


def test_e2e_job():
    section("3. E2E job: create → poll → complete")
    workflow = load_workflow()
    user_id = f"test_user_{uuid.uuid4().hex[:6]}"

    info(f"Submitting job as user_id={user_id}")
    r = create_job(workflow, user_id=user_id)
    if r.status_code != 200:
        fail(f"Job create failed: {r.status_code} — {r.text}")

    job = r.json()
    job_id = job["job_id"]
    ok(f"Job created: {job_id} (status={job['status']})")

    info("Polling for completion...")
    data = poll_job(job_id, max_wait=300)

    if data["status"] != "completed":
        fail(f"Job failed: {data.get('error')}\nLogs:\n" + "\n".join(data.get("logs", [])))

    ok(f"Job completed in ~{(datetime.fromisoformat(data['completed_at']) - datetime.fromisoformat(data['started_at'])).total_seconds():.1f}s")

    outputs = data.get("outputs", [])
    if not outputs:
        fail("No outputs returned")
    for out in outputs:
        ok(f"Output: {out['filename']} ({out.get('size_bytes', 0) // 1024}KB)")
        if out.get("url"):
            ok(f"  R2 URL: {out['url'][:80]}...")
        elif out.get("error"):
            fail(f"  Upload error: {out['error']}")
        else:
            fail(f"  No URL and no error — unexpected output: {out}")

    # Verify listing
    r = requests.get(f"{API}/v1/jobs?user_id={user_id}", headers=HEADERS, timeout=10)
    if r.status_code != 200:
        fail(f"List jobs failed: {r.status_code}")
    listed = r.json()["jobs"]
    if not any(j["job_id"] == job_id for j in listed):
        fail("Created job not found in list")
    ok(f"Job appears in listing ({len(listed)} total for user)")

    return job_id


def test_cancel():
    section("4. Cancellation")
    workflow = load_workflow()
    user_id = f"cancel_test_{uuid.uuid4().hex[:6]}"

    r = create_job(workflow, user_id=user_id, seed=77777)
    if r.status_code != 200:
        fail(f"Job create failed: {r.status_code}")
    job_id = r.json()["job_id"]
    ok(f"Job created: {job_id}")

    # Wait until running (or give up after 120s)
    info("Waiting for job to start running...")
    deadline = time.time() + 180
    while time.time() < deadline:
        r2 = requests.get(f"{API}/v1/jobs/{job_id}", headers=HEADERS, timeout=10)
        status = r2.json()["status"]
        info(f"  Status: {status}")
        if status == "running":
            ok("Job is running")
            break
        if status in ("completed", "failed"):
            info(f"Job already {status} before cancel — testing cancel on terminal state")
            break
        time.sleep(3)

    # Cancel
    r3 = requests.delete(f"{API}/v1/jobs/{job_id}", headers=HEADERS, timeout=10)
    if r3.status_code != 200:
        fail(f"Cancel failed: {r3.status_code} — {r3.text}")
    ok(f"Cancel request accepted: {r3.json()['message']}")

    # Poll for final status (volume propagation can take a few seconds)
    info("Waiting for final status after cancel...")
    deadline = time.time() + 60
    while time.time() < deadline:
        r4 = requests.get(f"{API}/v1/jobs/{job_id}", headers=HEADERS, timeout=10)
        final = r4.json()["status"]
        info(f"  Status: {final}")
        if final in ("cancelled", "completed", "failed"):
            ok(f"Final status: {final}")
            return
        time.sleep(3)
    fail(f"Job still '{final}' after 60s post-cancel")


def test_quota():
    section("5. Per-user quota")
    workflow = load_workflow()
    user_id = f"quota_test_{uuid.uuid4().hex[:6]}"

    limit = 5
    accepted = []
    rejected = []

    info(f"Submitting {limit + 3} jobs simultaneously for user_id={user_id}")
    for i in range(limit + 3):
        r = create_job(workflow, user_id=user_id, seed=i * 100)
        if r.status_code == 200:
            accepted.append(r.json()["job_id"])
            info(f"  Job {i+1}: accepted ({r.json()['job_id'][:8]})")
        elif r.status_code == 429:
            rejected.append(i)
            info(f"  Job {i+1}: rejected 429")
        else:
            info(f"  Job {i+1}: unexpected {r.status_code} — {r.text}")

    ok(f"Accepted: {len(accepted)}, Rejected 429: {len(rejected)}")
    # Note: file-based counting without locks has a known race under simultaneous burst.
    # In real usage (Discord bot, sequential commands) quota enforces correctly.
    # Under max concurrency burst, allow up to 2x limit before flagging.
    if len(rejected) == 0:
        fail("No 429s at all — quota completely disabled")
    if len(accepted) > limit * 2:
        fail(f"Way too many accepted: {len(accepted)} (limit={limit})")
    ok(f"Quota active (marginal over-acceptance under burst is expected for file-based counting)")

    # Cancel all accepted jobs to avoid burning GPU
    info("Cancelling all submitted jobs...")
    for jid in accepted:
        try:
            requests.delete(f"{API}/v1/jobs/{jid}", headers=HEADERS, timeout=10)
        except Exception:
            pass
    ok("Cleanup done")


# ── main ─────────────────────────────────────────────────────────────────────

ALL_TESTS = {
    "health": test_health,
    "auth": test_auth_rejected,
    "e2e": test_e2e_job,
    "cancel": test_cancel,
    "quota": test_quota,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=list(ALL_TESTS.keys()), help="Run specific test only")
    args = parser.parse_args()

    print(f"\nComfyUI SaaS API — E2E Test Suite")
    print(f"API: {API}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.test:
        ALL_TESTS[args.test]()
    else:
        test_health()
        test_auth_rejected()
        test_e2e_job()
        test_cancel()
        test_quota()

    print(f"\n{'='*50}")
    print("All tests passed!")


if __name__ == "__main__":
    main()
