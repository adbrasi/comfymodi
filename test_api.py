#!/usr/bin/env python3
"""
Test script for the ComfyUI SaaS API.

Usage:
    python test_api.py

Submits the SDXL workflow, polls for progress, and saves outputs.
"""

import json
import time
import base64
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration - adjust these for your deployment
# ---------------------------------------------------------------------------

# After deploying, get your URL from: modal app list
# Format: https://<workspace>--comfyui-saas-api.modal.run
API_URL = ""  # Will be auto-detected if empty
API_KEY = ""  # Set this to your API_KEY from Modal secrets

WORKFLOW_PATH = Path("workflows/sdxl_simple_exampleV2.json")
OUTPUT_DIR = Path("test_outputs")


def get_api_url() -> str:
    """Try to auto-detect the API URL from Modal."""
    if API_URL:
        return API_URL.rstrip("/")

    import subprocess
    try:
        result = subprocess.run(
            ["modal", "app", "list"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "comfyui-saas" in line and "deployed" in line.lower():
                # Extract URL from modal app list output
                print(f"Found app: {line.strip()}")
                break
    except Exception:
        pass

    # Fallback: ask user
    url = input("Enter your API URL (e.g. https://your-workspace--comfyui-saas-api.modal.run): ").strip()
    return url.rstrip("/")


def get_api_key() -> str:
    if API_KEY:
        return API_KEY
    key = input("Enter your API key: ").strip()
    return key


def main():
    base_url = get_api_url()
    api_key = get_api_key()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 1. Health check
    print("Checking API health...")
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        r.raise_for_status()
        print(f"  API is healthy: {r.json()}")
    except Exception as e:
        print(f"  Health check failed: {e}")
        print("  Make sure the API is deployed: modal deploy comfyui_api.py")
        sys.exit(1)

    # 2. Load workflow
    print(f"\nLoading workflow: {WORKFLOW_PATH}")
    if not WORKFLOW_PATH.exists():
        print(f"  ERROR: Workflow file not found: {WORKFLOW_PATH}")
        sys.exit(1)
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))

    # 3. Submit job
    print("\nSubmitting job...")
    payload = {
        "workflow": workflow,
        "inputs": [
            {
                "node": "6",
                "field": "text",
                "value": "masterpiece, best quality, 1girl, cherry blossoms, sunset, detailed background",
                "type": "raw",
            },
            {
                "node": "53",
                "field": "seed",
                "value": 42,
                "type": "raw",
            },
        ],
    }

    try:
        r = requests.post(f"{base_url}/v1/jobs", json=payload, headers=headers, timeout=30)
        if r.status_code == 401:
            print("  ERROR: Invalid API key")
            sys.exit(1)
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"  ERROR: {e}")
        print(f"  Response: {r.text}")
        sys.exit(1)

    job = r.json()
    job_id = job["job_id"]
    print(f"  Job ID: {job_id}")
    print(f"  Status: {job['status']}")

    # 4. Poll for completion
    print("\nWaiting for completion...")
    start = time.time()
    last_progress = -1

    while True:
        try:
            r = requests.get(f"{base_url}/v1/jobs/{job_id}", headers=headers, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(3)
            continue

        status = r.json()
        elapsed = time.time() - start
        progress = status.get("progress", 0)

        if progress != last_progress:
            bar_len = 30
            filled = int(bar_len * progress / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"\r  [{bar}] {progress:>3}% | {elapsed:.0f}s", end="", flush=True)
            last_progress = progress

        if status["status"] == "completed":
            print(f"\n\n  Completed in {elapsed:.1f}s!")

            # Save outputs
            OUTPUT_DIR.mkdir(exist_ok=True)
            outputs = status.get("outputs", [])
            print(f"  {len(outputs)} output(s):")

            for out in outputs:
                filename = out.get("filename", "output")
                size = out.get("size_bytes", 0)

                if out.get("url"):
                    # Download from R2
                    print(f"    {filename} ({size / 1024:.1f} KB) - downloading from R2...")
                    dr = requests.get(out["url"], timeout=60)
                    dr.raise_for_status()
                    out_path = OUTPUT_DIR / f"{job_id[:8]}_{filename}"
                    out_path.write_bytes(dr.content)
                    print(f"    Saved: {out_path}")

                elif out.get("data") and out.get("type") != "text":
                    # Base64 fallback
                    out_path = OUTPUT_DIR / f"{job_id[:8]}_{filename}"
                    out_path.write_bytes(base64.b64decode(out["data"]))
                    print(f"    Saved: {out_path} ({size / 1024:.1f} KB)")

                elif out.get("type") == "text":
                    print(f"    [text] {out.get('data', '')[:100]}")

            break

        elif status["status"] == "failed":
            print(f"\n\n  FAILED: {status.get('error', 'unknown error')}")
            sys.exit(1)

        if elapsed > 600:
            print("\n\n  Timeout (10 min)")
            sys.exit(1)

        time.sleep(3)

    print(f"\nDone! Outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
