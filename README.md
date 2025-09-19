# ComfyUI SaaS API on Modal

A production-ready, scalable ComfyUI API service deployed on Modal.com that transforms ComfyUI workflows into a powerful REST API. Perfect for building AI-powered image/video generation SaaS applications, creative tools, and automated content pipelines.

## 🚀 Features

- **Serverless GPU Infrastructure** - Automatic scaling with L40S GPUs on Modal
- **REST API** - Simple HTTP endpoints for job submission and management
- **Asynchronous Processing** - Non-blocking job queue with status tracking
- **Real-time Progress** - WebSocket-based progress updates with detailed metrics
- **Memory Snapshots** - Fast cold starts (under 10 seconds) with pre-loaded models
- **Multiple Output Types** - Supports images, videos, GIFs, audio, text, and JSON
- **Webhook Support** - Get notified when jobs complete
- **Cost Efficient** - Pay only for GPU time used, scales to zero when idle

## 📁 Project Structure

```
modal_novo/
├── modal_code2.py          # Main Modal deployment (recommended)
├── workflows/              # ComfyUI workflow JSON files
│   ├── image2image.json
│   └── image_to_video_wan21API.json
```

## 🏗️ Architecture

### Core Components

1. **ComfyService Class** - GPU-enabled container that runs ComfyUI
   - Handles workflow execution
   - Manages WebSocket connections
   - Tracks progress and outputs

2. **FastAPI Endpoints** - REST API for job management
   - `/v1/jobs` - Submit new jobs
   - `/v1/jobs/{job_id}` - Check job status
   - `/v1/jobs/{job_id}` (DELETE) - Cancel jobs

3. **Persistent Volumes**
   - `comfyui-cache` - Stores models (6GB+ for Animagine XL, WAN 2.1, etc.)
   - `job-storage` - Persists job data across containers

## 🛠️ Installation & Deployment

### Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **Python 3.8+**: Required for deployment
3. **Modal CLI**: Install with `pip install modal`

### Setup Steps

```bash
# 1. Clone the repository
git clone <your-repo>
cd modal_novo

# 2. Install Modal CLI
pip install modal

# 3. Authenticate with Modal
modal setup

# 4. Deploy the API
python -m modal deploy modal_code2.py

# 5. Your API will be available at:
# https://<your-username>--comfyui-saas-api-api.modal.run
```

### First-Time Setup

On first deployment, the system will:
1. Download and cache models (~20GB total)
2. Install ComfyUI and custom nodes
3. Create memory snapshots for fast starts

This initial setup takes 10-15 minutes but only happens once.

## 📡 API Documentation

### Authentication

Currently, the API is public. For production, add authentication headers:

```python
headers = {
    "Authorization": "Bearer YOUR_API_KEY"  # Implement in production
}
```

### Submit a Job

**POST** `/v1/jobs`

```python
import requests
import json
import base64

# Load your ComfyUI workflow
with open("workflow.json", "r") as f:
    workflow = json.load(f)

# Encode input image (if needed)
with open("input.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Submit job
response = requests.post(
    "https://your-api-url/v1/jobs",
    json={
        "workflow": workflow,
        "inputs": [
            {"name": "prompt", "value": "a beautiful landscape"},
            {"name": "negative_prompt", "value": "blurry, low quality"}
        ],
        "media": [
            {
                "name": "example.png",
                "data": image_base64  # Base64 encoded image
            }
        ],
        "webhook_url": "https://your-webhook.com/callback",  # Optional
        "priority": 1  # Optional, default 0
    }
)

job = response.json()
print(f"Job ID: {job['job_id']}")
print(f"Status: {job['status']}")
```

### Check Job Status

**GET** `/v1/jobs/{job_id}`

```python
response = requests.get(f"https://your-api-url/v1/jobs/{job_id}")
status = response.json()

print(f"Status: {status['status']}")  # queued, running, completed, failed
print(f"Progress: {status['progress']}%")

if status['status'] == 'completed':
    for output in status['outputs']:
        print(f"Output: {output['filename']}")
        # Decode base64 image/video
        output_data = base64.b64decode(output['data'])
        with open(output['filename'], 'wb') as f:
            f.write(output_data)
```

### Cancel a Job

**DELETE** `/v1/jobs/{job_id}`

```python
response = requests.delete(f"https://your-api-url/v1/jobs/{job_id}")
```

## 🎨 Workflow Integration

### Supported Workflows

The API supports any ComfyUI workflow. Common use cases:

1. **Image Generation** (Stable Diffusion, SDXL, Animagine XL)
2. **Image-to-Image** (Style transfer, img2img)
3. **Video Generation** (WAN 2.1, AnimateDiff)
4. **Upscaling** (ESRGAN, Real-ESRGAN)
5. **ControlNet** (Pose, depth, canny edge)

### Workflow Requirements

Your workflow JSON must:
1. Use `LoadImage` nodes with names matching your media inputs
2. Use `SaveImage` or similar nodes for outputs
3. Have text inputs in `CLIPTextEncode` nodes

### Example: Image-to-Image Workflow

```json
{
  "6": {
    "inputs": {
      "text": "YOUR_PROMPT_HERE",
      "clip": ["14", 1]
    },
    "class_type": "CLIPTextEncode"
  },
  "10": {
    "inputs": {
      "image": "example.png"  // Matches media[].name
    },
    "class_type": "LoadImage"
  }
}
```

## 💡 Building Applications

### SaaS Platform Example

```python
# app.py - Flask/FastAPI backend
from flask import Flask, request, jsonify
import requests
import stripe

app = Flask(__name__)
COMFY_API = "https://your-modal-api.run"

@app.post("/api/generate")
async def generate_image():
    user = authenticate(request)

    # Check credits/subscription
    if not user.has_credits():
        return {"error": "Insufficient credits"}, 402

    # Submit to ComfyUI API
    job = requests.post(
        f"{COMFY_API}/v1/jobs",
        json={
            "workflow": load_workflow("sdxl"),
            "inputs": request.json,
            "webhook_url": f"https://yourapp.com/webhook/{user.id}"
        }
    ).json()

    # Deduct credits
    user.use_credits(1)

    return {"job_id": job["job_id"]}

@app.post("/webhook/<user_id>")
async def handle_completion(user_id):
    # Process completed job
    result = request.json
    save_to_user_gallery(user_id, result["outputs"])
    notify_user(user_id, "Your image is ready!")
```

### React Frontend Integration

```jsx
// GenerateImage.jsx
import { useState } from 'react';

function GenerateImage() {
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);

  const generate = async (prompt) => {
    setLoading(true);

    // Submit job
    const job = await fetch('/api/generate', {
      method: 'POST',
      body: JSON.stringify({ prompt })
    }).then(r => r.json());

    // Poll for status
    const interval = setInterval(async () => {
      const status = await fetch(`/api/jobs/${job.job_id}`)
        .then(r => r.json());

      setProgress(status.progress);

      if (status.status === 'completed') {
        clearInterval(interval);
        setResult(status.outputs[0].data);
        setLoading(false);
      }
    }, 2000);
  };

  return (
    <div>
      <button onClick={() => generate("a cat in space")}>
        Generate
      </button>
      {loading && <ProgressBar value={progress} />}
      {result && <img src={`data:image/png;base64,${result}`} />}
    </div>
  );
}
```

### Discord Bot Example

```python
# discord_bot.py
import discord
from discord.ext import commands
import requests
import asyncio

bot = commands.Bot(command_prefix='!')
COMFY_API = "https://your-modal-api.run"

@bot.command()
async def imagine(ctx, *, prompt):
    """Generate an image from a text prompt"""

    # Submit job
    response = requests.post(
        f"{COMFY_API}/v1/jobs",
        json={
            "workflow": load_workflow("sdxl"),
            "inputs": [{"name": "prompt", "value": prompt}]
        }
    )
    job = response.json()

    # Send initial message
    msg = await ctx.send(f"🎨 Generating... (0%)")

    # Poll for completion
    while True:
        await asyncio.sleep(2)

        status = requests.get(
            f"{COMFY_API}/v1/jobs/{job['job_id']}"
        ).json()

        # Update progress
        await msg.edit(content=f"🎨 Generating... ({status['progress']}%)")

        if status['status'] == 'completed':
            # Send image
            image_data = base64.b64decode(status['outputs'][0]['data'])
            file = discord.File(io.BytesIO(image_data), 'generated.png')
            await ctx.send(file=file)
            await msg.delete()
            break
```

## 🚀 Advanced Features

### Batch Processing

```python
# Process multiple images in parallel
jobs = []
for image in images:
    response = requests.post(
        f"{COMFY_API}/v1/jobs",
        json={"workflow": workflow, "media": [{"name": "input.png", "data": image}]}
    )
    jobs.append(response.json()['job_id'])

# Wait for all to complete
results = []
for job_id in jobs:
    while True:
        status = requests.get(f"{COMFY_API}/v1/jobs/{job_id}").json()
        if status['status'] in ['completed', 'failed']:
            results.append(status)
            break
        time.sleep(1)
```

### Webhook Processing

```python
@app.post("/comfyui-webhook")
async def process_webhook(request):
    data = await request.json()

    if data['status'] == 'completed':
        # Process outputs
        for output in data['outputs']:
            if output['type'] == 'image':
                save_image(output['data'])
            elif output['type'] == 'video':
                process_video(output['data'])
            elif output['type'] == 'text':
                save_metadata(output['data'])

    return {"status": "ok"}
```

### Priority Queue

```python
# High priority job (processed first)
high_priority_job = {
    "workflow": workflow,
    "priority": 10,  # Higher = processed sooner
    "inputs": premium_user_inputs
}

# Low priority job
low_priority_job = {
    "workflow": workflow,
    "priority": 1,
    "inputs": free_user_inputs
}
```

## 📊 Performance & Scaling

### Performance Metrics

- **Cold Start**: ~8-10 seconds (with memory snapshots)
- **Warm Start**: Instant (container already running)
- **Image Generation**: 3-15 seconds (depending on resolution)
- **Video Generation**: 30-120 seconds (depending on length)
- **Concurrent Jobs**: 5 per container (configurable)
- **Max Containers**: 10 (configurable, can scale higher)

### Cost Optimization

```python
# modal_code2.py configuration
@app.cls(
    gpu="L40S",              # $0.00097/second (~$3.50/hour)
    scaledown_window=300,    # Keep alive for 5 minutes
    max_containers=10,       # Max concurrent containers
)
@modal.concurrent(max_inputs=5)  # Process 5 jobs per container
```

### Monitoring

```python
# Get system stats
response = requests.get("https://your-api-url/health")
print(response.json())
# {
#   "status": "healthy",
#   "active_jobs": 5,
#   "queued_jobs": 12,
#   "gpu_utilization": 85,
#   "memory_usage": "12GB/24GB"
# }
```

## 🔧 Customization

### Adding New Models

Edit `modal_code2.py`:

```python
def download_assets_and_setup():
    model_files = [
        # Add your model
        ("huggingface-user/model-name",
         "model.safetensors",
         f"{CACHE_DIR}/models/checkpoints"),
    ]
```

### Custom Nodes

```python
# In modal_code2.py, add to install_custom_nodes()
"cd /root/comfy/ComfyUI/custom_nodes && git clone https://github.com/your/custom-node"
```

### GPU Configuration

```python
# For different GPU types
gpu="A100"      # Faster but more expensive
gpu="L40S"      # Good balance (default)
gpu="A10G"      # Budget option
```

## 🐛 Troubleshooting

### Common Issues

1. **"No CUDA GPUs available"**
   - The snapshot is being created without GPU
   - Solution: Already fixed in modal_code2.py

2. **"Job not found"**
   - Volume sync issue
   - Solution: Code includes `job_volume.reload()`

3. **WebSocket decode errors**
   - Binary preview frames
   - Solution: Proper handling added for binary messages

4. **Slow cold starts**
   - Models loading from scratch
   - Solution: Memory snapshots enabled

## 📈 Production Checklist

- [ ] Add authentication (API keys, JWT)
- [ ] Implement rate limiting
- [ ] Add request validation
- [ ] Set up monitoring (Datadog, New Relic)
- [ ] Configure alerts for failures
- [ ] Add database for job history
- [ ] Implement cleanup for old jobs
- [ ] Set up CDN for output files
- [ ] Add CORS configuration
- [ ] Implement request signing for webhooks

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- ComfyUI for the amazing workflow system
- Modal.com for serverless GPU infrastructure
- The AI art community for workflows and models