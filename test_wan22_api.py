#!/usr/bin/env python3
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
