#!/usr/bin/env python3
"""Quick test run with detailed progress display and logging.

Usage:
    export COMFYUI_API_URL=https://yourworkspace--comfyui-saas-api.modal.run
    export COMFYUI_API_KEY=your-api-key
    python test_run.py
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

API = os.environ.get("COMFYUI_API_URL", "https://cezarsaint--comfyui-saas-api.modal.run")
KEY = os.environ.get("COMFYUI_API_KEY", "")

if not KEY:
    print("ERROR: set COMFYUI_API_KEY env var")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {KEY}",
}

workflow = json.loads(Path("workflows/sdxl_simple_exampleV2.json").read_text())

print("=== ComfyUI SDXL Test Run ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"API:  {API}")
print()

r = requests.post(
    f"{API}/v1/jobs",
    json={
        "workflow": workflow,
        "inputs": [
            {"node": "6", "field": "text", "value": "cinematic photo, beautiful woman, sunset, film grain", "type": "raw"},
            {"node": "53", "field": "seed", "value": 999, "type": "raw"},
            {"node": "53", "field": "steps", "value": 20, "type": "raw"},
        ],
        "user_id": "test_user_123",
    },
    headers=HEADERS,
    timeout=30,
)
job = r.json()
job_id = job["job_id"]
print(f"Job ID:  {job_id}")
print(f"Status:  {job['status']}")
print()

start = time.time()
last_progress = -1
last_node = None
last_step = -1

while True:
    try:
        r = requests.get(f"{API}/v1/jobs/{job_id}", headers=HEADERS, timeout=10)
        s = r.json()
    except Exception as e:
        print(f"Poll error: {e}")
        time.sleep(3)
        continue

    elapsed = time.time() - start
    progress = s.get("progress", 0)
    status = s["status"]
    node = s.get("current_node")
    step = s.get("current_step", 0)
    total_step = s.get("total_steps", 0)
    nodes_done = s.get("nodes_done", 0)
    nodes_total = s.get("nodes_total", 0)

    changed = progress != last_progress or node != last_node or step != last_step
    if changed:
        bar = "█" * int(progress / 5) + "░" * (20 - int(progress / 5))
        step_str = f" | step {step}/{total_step}" if total_step > 0 else ""
        node_str = f" | node {node}" if node else ""
        nodes_str = f" | {nodes_done}/{nodes_total} nodes" if nodes_total > 0 else ""
        print(f"[{elapsed:5.0f}s] [{bar}] {progress:3}%{node_str}{step_str}{nodes_str}  ({status})")
        last_progress = progress
        last_node = node
        last_step = step

    if status == "completed":
        print()
        print("=== JOB COMPLETED ===")
        print(f"Total time: {elapsed:.1f}s")
        print()
        print("--- Execution Logs ---")
        for line in s.get("logs", []):
            print(f"  {line}")
        print()
        print("--- Outputs ---")
        for out in s.get("outputs", []):
            if out.get("url"):
                print(f"  [{out['type']}] {out['filename']} ({out['size_bytes']//1024}KB)")
                print(f"  URL: {out['url'][:90]}...")
            elif out.get("type") == "text":
                print(f"  [text] {out['data'][:100]}")
        break

    if status in ("failed", "cancelled"):
        print()
        print(f"=== JOB {status.upper()} ===")
        print(f"Error: {s.get('error')}")
        print()
        print("--- Logs ---")
        for line in s.get("logs", []):
            print(f"  {line}")
        sys.exit(1)

    time.sleep(2)
