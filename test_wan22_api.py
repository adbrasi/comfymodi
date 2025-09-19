#!/usr/bin/env python3
"""
Simple test script for the Wan22 Smooth Loop workflow.
"""

import requests
import json
import base64
import time
from pathlib import Path

# Your Modal API endpoint
API_URL = "https://cezarsaint--comfyui-saas-api-api.modal.run"

def run_workflow(image_path: str, prompt: str):
    """Quick function to run the workflow."""

    # Load workflow (with UTF-8 encoding for Windows compatibility)
    with open("workflows\wan22Smoothloop_fixed.json", "r", encoding="utf-8") as f:
        workflow = json.load(f)

    # Encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    image_filename = Path(image_path).name

    # Update workflow to use our image
    workflow["107"]["inputs"]["image"] = image_filename

    # Submit job
    print(f"🚀 Submitting job with image: {image_filename}")
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
        }
    )

    if response.status_code != 200:
        print(f"❌ Error: {response.text}")
        return None

    job = response.json()
    job_id = job['job_id']
    print(f"✅ Job ID: {job_id}")

    # Poll for status
    print("⏳ Processing... (this takes 30 seconds to 2 minutes)")
    for i in range(120):  # Check for up to 6 minutes
        status = requests.get(f"{API_URL}/v1/jobs/{job_id}").json()

        if status['status'] == 'completed':
            print(f"\n✨ Completed!")

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
            return status

        # Show progress
        progress = status.get('progress', 0)
        print(f"\rProgress: {progress}%", end="")

        time.sleep(3)

    print("\n⏱️ Timeout!")
    return None

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

    # Run the workflow
    result = run_workflow(
        image_path=test_image,
        prompt="IntenseAnimation, a dynamic and smooth animation with a lot of movement, looping. colorful, vibrant."
    )

    if result:
        print("\n🎉 Done! Check the output files.")