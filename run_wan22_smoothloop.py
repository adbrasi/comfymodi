#!/usr/bin/env python3
"""
Run the Wan22 Smooth Loop workflow using the Modal API.
This workflow creates animated loops from images.
"""

import requests
import json
import base64
import time
import sys
from pathlib import Path
from typing import Optional

# API Configuration
API_URL = "https://cezarsaint--comfyui-saas-api-api.modal.run"

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def submit_job(
    prompt: str,
    image_path: str,
    negative_prompt: Optional[str] = None,
    webhook_url: Optional[str] = None
):
    """Submit a Wan22 Smooth Loop job to the API."""

    # Load the workflow
    workflow_path = Path("workflows/wan22Smoothloop.json")
    if not workflow_path.exists():
        print(f"❌ Workflow file not found: {workflow_path}")
        sys.exit(1)

    with open(workflow_path, "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Verify the image exists
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        sys.exit(1)

    # Encode the image
    print(f"📸 Encoding image: {image_path}")
    image_base64 = encode_image_to_base64(image_path)
    image_filename = Path(image_path).name

    # Prepare the inputs
    inputs = [
        {
            "node": "16",
            "field": "positive_prompt",
            "value": prompt,
            "type": "raw"
        }
    ]

    # Add negative prompt if provided
    if negative_prompt:
        inputs.append({
            "node": "16",
            "field": "negative_prompt",
            "value": negative_prompt,
            "type": "raw"
        })

    # Prepare media files
    media = [
        {
            "name": image_filename,  # This should match what's in node 107
            "data": image_base64
        }
    ]

    # Update workflow to use the correct filename
    if "107" in workflow:
        workflow["107"]["inputs"]["image"] = image_filename

    # Prepare the request
    request_data = {
        "workflow": workflow,
        "inputs": inputs,
        "media": media,
        "priority": 1
    }

    if webhook_url:
        request_data["webhook_url"] = webhook_url

    # Submit the job
    print("🚀 Submitting job to API...")
    try:
        response = requests.post(
            f"{API_URL}/v1/jobs",
            json=request_data,
            timeout=30
        )
        response.raise_for_status()
        job_data = response.json()
        print(f"✅ Job submitted successfully!")
        print(f"📋 Job ID: {job_data['job_id']}")
        return job_data['job_id']
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to submit job: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        sys.exit(1)

def check_job_status(job_id: str):
    """Check the status of a job."""
    try:
        response = requests.get(
            f"{API_URL}/v1/jobs/{job_id}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to check job status: {e}")
        return None

def wait_for_completion(job_id: str, max_wait: int = 300):
    """Wait for a job to complete and download results."""
    print(f"\n⏳ Waiting for job to complete (max {max_wait} seconds)...")
    print("This workflow typically takes 30 seconds to 2 minutes.")

    start_time = time.time()
    last_progress = -1

    while time.time() - start_time < max_wait:
        status = check_job_status(job_id)

        if not status:
            time.sleep(5)
            continue

        # Show progress if it changed
        current_progress = status.get('progress', 0)
        if current_progress != last_progress:
            print(f"📊 Progress: {current_progress}%", end="")
            if status.get('progress_details'):
                details = status['progress_details']
                if details.get('current_node'):
                    print(f" - Node: {details['current_node']}", end="")
                if details.get('step'):
                    print(f" - Step: {details['step']}", end="")
            print()  # New line
            last_progress = current_progress

        if status['status'] == 'completed':
            print(f"\n✨ Job completed successfully!")
            elapsed = time.time() - start_time
            print(f"⏱️ Time taken: {elapsed:.1f} seconds")

            # Save outputs
            if status.get('outputs'):
                print(f"\n📦 Found {len(status['outputs'])} outputs:")

                # Create output directory
                output_dir = Path("outputs") / job_id
                output_dir.mkdir(parents=True, exist_ok=True)

                for i, output in enumerate(status['outputs']):
                    try:
                        filename = output.get('filename', f'output_{i}')
                        file_type = output.get('type', 'video')

                        # Ensure proper extension for videos
                        if file_type == 'video' and not filename.endswith(('.mp4', '.webm', '.avi', '.mov')):
                            filename = f"{Path(filename).stem}.mp4"

                        output_path = output_dir / filename

                        # Decode base64 data (works for videos, images, any binary data)
                        if output.get('data'):
                            data = output['data']

                            # Handle potential padding issues
                            try:
                                file_data = base64.b64decode(data)
                            except:
                                # Try adding padding if needed
                                missing_padding = len(data) % 4
                                if missing_padding:
                                    data += '=' * (4 - missing_padding)
                                file_data = base64.b64decode(data)

                            with open(output_path, 'wb') as f:
                                f.write(file_data)

                            size_mb = len(file_data) / (1024 * 1024)
                            file_type_str = "video" if file_type == 'video' else "file"
                            print(f"  💾 Saved {file_type_str}: {output_path} ({size_mb:.2f} MB)")
                    except Exception as e:
                        print(f"  ⚠️ Could not save output {i}: {e}")

                print(f"\n✅ All outputs saved to: {output_dir}")
            else:
                print("⚠️ No outputs found in the response")

            return status

        elif status['status'] == 'failed':
            print(f"\n❌ Job failed: {status.get('error', 'Unknown error')}")
            return status

        # Wait before next check
        time.sleep(3)

    print(f"\n⏱️ Timeout: Job did not complete within {max_wait} seconds")
    return None

def main():
    """Main function to run the workflow."""
    print("🎬 Wan22 Smooth Loop Workflow Runner")
    print("=" * 50)

    # Example usage - you can modify these parameters
    prompt = "IntenseAnimation, a dynamic and smooth animation with a lot of movement, looping. love, heart."

    # You can change this to your image path
    image_path = "Fubuki 022.jpg"  # Make sure this file exists

    # Optional: Custom negative prompt (default is already in the workflow)
    negative_prompt = None

    # Check if image exists, if not use a placeholder message
    if not Path(image_path).exists():
        print(f"\n⚠️ Image '{image_path}' not found!")
        print("\nTo use this script:")
        print("1. Place your input image in the current directory")
        print("2. Update the 'image_path' variable in the script")
        print("3. Optionally modify the 'prompt' variable")
        print("\nExample usage in the script:")
        print('  prompt = "Your animation prompt here"')
        print('  image_path = "your_image.jpg"')
        return

    # Submit the job
    job_id = submit_job(
        prompt=prompt,
        image_path=image_path,
        negative_prompt=negative_prompt
    )

    # Wait for completion and get results
    result = wait_for_completion(job_id, max_wait=300)  # 5 minutes max

    if result and result['status'] == 'completed':
        print("\n🎉 Workflow completed successfully!")
        print(f"Check the outputs directory for your animated loop.")
    else:
        print("\n😞 Workflow did not complete successfully.")

if __name__ == "__main__":
    # You can also run with command line arguments if desired
    import argparse

    parser = argparse.ArgumentParser(description="Run Wan22 Smooth Loop workflow")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--prompt", type=str, help="Animation prompt")
    parser.add_argument("--negative", type=str, help="Negative prompt (optional)")
    parser.add_argument("--webhook", type=str, help="Webhook URL for notifications (optional)")

    args = parser.parse_args()

    if args.image and args.prompt:
        # Run with command line arguments
        job_id = submit_job(
            prompt=args.prompt,
            image_path=args.image,
            negative_prompt=args.negative,
            webhook_url=args.webhook
        )

        result = wait_for_completion(job_id, max_wait=300)

        if result and result['status'] == 'completed':
            print("\n🎉 Success!")
        else:
            print("\n❌ Failed!")
    else:
        # Run with hardcoded values
        main()