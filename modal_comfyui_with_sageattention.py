import json
import uuid
import base64
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

import modal
import requests
import websocket
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

APP_NAME = "comfyui-saas-apiLASTDANCE2"
app = modal.App(APP_NAME)

# CRITICAL: Single cache volume mounted at EMPTY path
cache_volume = modal.Volume.from_name("comfyui-cache", create_if_missing=True)
CACHE_DIR = "/cache"

# Job storage volume (persistent across deployments)
job_volume = modal.Volume.from_name("job-storage", create_if_missing=True)
JOB_DIR = "/jobs"

# ComfyUI paths (these will be symlinks)
COMFY_DIR = Path("/root/comfy/ComfyUI")
INPUT_DIR = COMFY_DIR / "input"

# Job status enum
class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# API Models
class MediaFile(BaseModel):
    name: str
    data: Optional[str] = None  # base64
    url: Optional[str] = None

class WorkflowInput(BaseModel):
    node: str
    field: str
    value: Any
    type: str = "raw"

class JobRequest(BaseModel):
    workflow: Dict
    inputs: Optional[List[WorkflowInput]] = []
    media: Optional[List[MediaFile]] = []
    webhook_url: Optional[str] = None
    priority: int = 0

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    estimated_time: Optional[int] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: int = 0
    outputs: List[Dict] = []
    error: Optional[str] = None

def download_assets_and_setup():
    """Download model files during image build"""
    from huggingface_hub import hf_hub_download
    
    print("🚀 Setting up ComfyUI models...")
    
    # Create directory structure
    for subdir in ["models/clip_vision", "models/diffusion_models", 
                   "models/text_encoders", "models/vae", "models/checkpoints",
                   "models/loras", "models/controlnet", "models/upscale_models",
                   "outputs", "temp"]:
        Path(f"{CACHE_DIR}/{subdir}").mkdir(parents=True, exist_ok=True)
    
    # Download models using HF Transfer
    model_files = [
        # Your existing model list...
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", 
         "split_files/clip_vision/clip_vision_h.safetensors", 
         f"{CACHE_DIR}/models/clip_vision"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged", 
         "split_files/diffusion_models/wan2.1_i2v_720p_14B_fp8_scaled.safetensors",
         f"{CACHE_DIR}/models/diffusion_models"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
         "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
         f"{CACHE_DIR}/models/text_encoders"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
         "split_files/text_encoders/umt5_xxl_fp16.safetensors",
         f"{CACHE_DIR}/models/text_encoders"),
        ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
         "split_files/vae/wan_2.1_vae.safetensors",
         f"{CACHE_DIR}/models/vae"),
        ("cagliostrolab/animagine-xl-4.0",
         "animagine-xl-4.0.safetensors",
         f"{CACHE_DIR}/models/checkpoints"),
    ]
    
    for repo_id, filename, local_dir in model_files:
        try:
            target_path = Path(local_dir) / Path(filename).name
            if target_path.exists() and target_path.stat().st_size > 1000000:  # Check if > 1MB
                print(f"✅ {filename} already exists, skipping...")
                continue
            
            print(f"📥 Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✅ Downloaded {filename}")
        except Exception as e:
            print(f"⚠️ Failed to download {filename}: {e}")
            continue
    
    cache_volume.commit()
    print("✅ Models committed to volume!")

def install_custom_nodes():
    """Install custom nodes with proper error handling"""
    import subprocess
    
    nodes_dir = Path("/root/comfy/ComfyUI/custom_nodes")
    
    custom_nodes = [
        "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
        "https://github.com/ltdrdata/ComfyUI-Manager",
        "https://github.com/WASasquatch/was-node-suite-comfyui",
        "https://github.com/cubiq/ComfyUI_essentials",
        "https://github.com/kijai/ComfyUI-KJNodes",
    ]
    
    for node_url in custom_nodes:
        node_name = node_url.split("/")[-1]
        node_path = nodes_dir / node_name
        
        try:
            if not node_path.exists():
                print(f"📦 Installing {node_name}...")
                subprocess.run([
                    "git", "clone", "--depth", "1", node_url, str(node_path)
                ], check=True, timeout=60)
                
                # Install requirements if exists
                req_file = node_path / "requirements.txt"
                if req_file.exists():
                    subprocess.run([
                        "pip", "install", "-q", "-r", str(req_file)
                    ], check=False, timeout=120)
                
                print(f"✅ Installed {node_name}")
            else:
                print(f"✅ {node_name} already exists")
        except Exception as e:
            print(f"⚠️ Failed to install {node_name}: {e}")

# Build the image
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "wget", "curl", "libgl1", "libglib2.0-0", "ffmpeg")
    # Install PyTorch with CUDA
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    # Install dependencies
    .pip_install(
        "comfy-cli==1.4.1",
        "huggingface_hub[hf_transfer]==0.34.4",
        "fastapi[standard]",
        "websocket-client",
        "pillow",
        "opencv-python-headless",
        "httpx",
        "triton>=3.0.0",
        "https://huggingface.co/adbrasi/comfywheel/resolve/main/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl",
        "aiofiles",  # For async file operations
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",  # Better for logging
    })
    # Install ComfyUI
    .run_commands(
        "cd /root && comfy --skip-prompt install --fast-deps --nvidia",
    )
    # Install custom nodes during build
    .run_function(install_custom_nodes)
    # Setup symlinks
    .run_commands(
        "rm -rf /root/comfy/ComfyUI/models",
        "rm -rf /root/comfy/ComfyUI/output",
        f"ln -s {CACHE_DIR}/models /root/comfy/ComfyUI/models",
        f"ln -s {CACHE_DIR}/outputs /root/comfy/ComfyUI/output",
        "mkdir -p /root/comfy/ComfyUI/input",
        "mkdir -p /root/comfy/ComfyUI/temp",
    )
    # Download models
    .run_function(
        download_assets_and_setup,
        volumes={CACHE_DIR: cache_volume},
        timeout=3600
    )
)

@app.cls(
    gpu="L40S",
    image=image,
    volumes={
        CACHE_DIR: cache_volume,
        JOB_DIR: job_volume
    },
    enable_memory_snapshot=True,  # CRITICAL: Re-enable for 10x speed
    scaledown_window=100,
    max_containers=1,
)
@modal.concurrent(max_inputs=5)
class ComfyService:
    
    @modal.enter(snap=True)  # Snapshot WITHOUT starting ComfyUI
    def setup_environment(self):
        """Setup environment and preload libraries - GPU not available during snapshot"""
        import sys
        import os
        
        print("📸 Creating memory snapshot (no GPU available yet)...")
        
        # Pre-import heavy Python libraries that are already installed
        # These imports will be cached in the snapshot
        import torch  # Already installed
        import numpy  # Installed as torch dependency
        import PIL  # From pillow package
        import cv2  # From opencv-python-headless
        # Don't import transformers/safetensors/diffusers - not installed
        
        # Set environment variables
        os.environ["COMFYUI_PATH"] = "/root/comfy/ComfyUI"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # Verify resources are accessible (but don't start ComfyUI)
        models_count = len(list(Path(f"{CACHE_DIR}/models").rglob("*.safetensors")))
        print(f"📊 Found {models_count} model files")
        
        # Verify custom nodes are installed
        custom_nodes_path = Path("/root/comfy/ComfyUI/custom_nodes")
        if custom_nodes_path.exists():
            nodes_count = len([d for d in custom_nodes_path.iterdir() if d.is_dir()])
            print(f"📦 Found {nodes_count} custom nodes")
        
        print("✅ Environment snapshot created (ComfyUI will start when GPU is available)")
        
        # Initialize process variable
        self.process = None
    
    @modal.enter(snap=False)  # Run AFTER snapshot restore, when GPU is available
    def start_comfy_with_gpu(self):
        """Start ComfyUI server with retry logic when GPU is available"""
        import subprocess
        import time
        
        print("🎮 GPU now available, starting ComfyUI server...")
        
        # Only start if not already running
        if self.process is None:
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    print(f"📡 Starting ComfyUI server (attempt {attempt + 1}/{max_retries})...")
                    
                    # Start ComfyUI with GPU available
                    self.process = subprocess.Popen([
                        "python", "/root/comfy/ComfyUI/main.py",
                        "--listen", "0.0.0.0",
                        "--port", "8188",
                        "--use-sage-attention",
                        "--disable-auto-launch",
                        # Removed --preview-method none to allow previews
                    ])
                    
                    # Wait for server to be ready
                    server_ready = False
                    start_time = time.time()
                    
                    while time.time() - start_time < 60:
                        try:
                            response = requests.get("http://localhost:8188/system_stats", timeout=2)
                            if response.status_code == 200:
                                print("✅ ComfyUI server ready with GPU!")
                                server_ready = True
                                break
                        except:
                            time.sleep(2)
                    
                    if server_ready:
                        break  # Success, exit retry loop
                    else:
                        # Kill the process if it didn't start properly
                        if self.process:
                            self.process.terminate()
                            self.process.wait(timeout=5)
                            self.process = None
                        
                        if attempt < max_retries - 1:
                            print(f"⚠️ Server failed to start, retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            raise Exception(f"ComfyUI server failed to start after {max_retries} attempts")
                            
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise Exception(f"Failed to start ComfyUI after {max_retries} attempts: {e}")
                    print(f"⚠️ Attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(retry_delay)
        else:
            print("♻️ ComfyUI server already running from previous container!")
    
    @modal.method()
    def process_job(self, job_id: str):
        """Process a ComfyUI job with optimized progress tracking"""
        progress_update_counter = 0
        
        try:
            # Load job (reload volume only once at start)
            job_volume.reload()
            job_file = Path(JOB_DIR) / f"{job_id}.json"
            with open(job_file, 'r') as f:
                job_data = json.load(f)
            
            # Update status to running
            job_data["status"] = "running"
            job_data["started_at"] = datetime.now(timezone.utc).isoformat()
            with open(job_file, 'w') as f:
                json.dump(job_data, f)
            job_volume.commit()
            
            # Process workflow
            workflow = job_data["workflow"]
            inputs = job_data.get("inputs", [])
            media = job_data.get("media", [])
            
            # Handle media uploads
            for item in media:
                if item.get("data"):
                    file_data = base64.b64decode(item["data"])
                    file_path = INPUT_DIR / item["name"]
                    file_path.write_bytes(file_data)
                elif item.get("url"):
                    response = requests.get(item["url"], timeout=30)
                    file_path = INPUT_DIR / item["name"]
                    file_path.write_bytes(response.content)
            
            # Apply dynamic inputs
            for inp in inputs:
                node_id = str(inp["node"])
                if node_id in workflow:
                    if "inputs" not in workflow[node_id]:
                        workflow[node_id]["inputs"] = {}
                    
                    field = inp["field"]
                    value = inp["value"]
                    
                    if inp.get("type") == "image_base64":
                        img_data = base64.b64decode(value)
                        filename = f"input_{node_id}_{field}.png"
                        (INPUT_DIR / filename).write_bytes(img_data)
                        workflow[node_id]["inputs"][field] = filename
                    elif inp.get("type") == "image_url":
                        response = requests.get(value, timeout=30)
                        filename = f"input_{node_id}_{field}.png"
                        (INPUT_DIR / filename).write_bytes(response.content)
                        workflow[node_id]["inputs"][field] = filename
                    else:
                        workflow[node_id]["inputs"][field] = value
            
            # Execute via WebSocket with better error handling
            client_id = str(uuid.uuid4())
            ws = websocket.WebSocket()
            ws.connect(f"ws://localhost:8188/ws?clientId={client_id}")
            
            try:
                # Submit prompt
                response = requests.post(
                    "http://localhost:8188/prompt",
                    json={"prompt": workflow, "client_id": client_id},
                    timeout=30
                )
                
                if response.status_code != 200:
                    raise Exception(f"Failed to submit workflow: {response.text}")
                
                prompt_id = response.json()["prompt_id"]
                print(f"📋 Executing prompt {prompt_id}")
                
                # Improved progress tracking with all message types
                progress_data = {
                    "nodes_total": len(workflow),
                    "nodes_completed": 0,
                    "current_node": None,
                    "ksampler_progress": 0,
                    "previews": []
                }
                last_commit_time = time.time()
                commit_interval = 10  # Commit every 10 seconds max
                
                while True:
                    try:
                        raw_msg = ws.recv()
                        
                        # Handle binary preview frames properly
                        if isinstance(raw_msg, bytes):
                            # Store preview if enabled (optional)
                            if job_data.get("enable_preview", False):
                                preview_data = base64.b64encode(raw_msg).decode()
                                progress_data["previews"].append(preview_data)
                                # Keep only last 3 previews to avoid memory issues
                                if len(progress_data["previews"]) > 3:
                                    progress_data["previews"].pop(0)
                            continue
                        
                        msg = json.loads(raw_msg)
                        msg_type = msg.get("type")
                        
                        if msg_type == "execution_start":
                            print(f"🚀 Starting execution for prompt {prompt_id}")
                        
                        elif msg_type == "execution_cached":
                            # Nodes that used cached results
                            cached_nodes = msg.get("data", {}).get("nodes", [])
                            progress_data["nodes_completed"] += len(cached_nodes)
                            print(f"⚡ {len(cached_nodes)} nodes using cache")
                        
                        elif msg_type == "executing":
                            node = msg.get("data", {}).get("node")
                            if node is None:
                                print("✅ Execution complete!")
                                break
                            progress_data["current_node"] = node
                            progress_data["nodes_completed"] += 1
                            print(f"🔧 Executing node {node} ({progress_data['nodes_completed']}/{progress_data['nodes_total']})")
                        
                        elif msg_type == "progress":
                            # KSampler or other node progress
                            data = msg.get("data", {})
                            current = data.get("value", 0)
                            total = max(data.get("max", 1), 1)
                            progress_data["ksampler_progress"] = current / total
                            
                            # Calculate overall progress
                            node_progress = progress_data["nodes_completed"] / progress_data["nodes_total"]
                            ksampler_weight = 0.5  # KSampler typically takes 50% of time
                            
                            if progress_data["ksampler_progress"] > 0:
                                overall = (node_progress * (1 - ksampler_weight)) + (progress_data["ksampler_progress"] * ksampler_weight)
                            else:
                                overall = node_progress
                            
                            job_data["progress"] = int(overall * 100)
                            job_data["progress_details"] = {
                                "current_node": progress_data["current_node"],
                                "nodes_completed": progress_data["nodes_completed"],
                                "nodes_total": progress_data["nodes_total"],
                                "step": f"{current}/{total}"
                            }
                            print(f"📊 Progress: {job_data['progress']}% (Step {current}/{total})")
                        
                        elif msg_type == "executed":
                            # Node completed with output
                            node_id = msg.get("data", {}).get("node")
                            output = msg.get("data", {}).get("output", {})
                            print(f"✓ Node {node_id} completed")
                        
                        elif msg_type == "execution_error":
                            error_data = msg.get("data", {})
                            raise Exception(f"Execution error: {error_data}")
                        
                        elif msg_type == "status":
                            # Status updates, can be logged if needed
                            pass
                        
                        # Batch commit - only commit if enough time has passed
                        if time.time() - last_commit_time > commit_interval:
                            with open(job_file, 'w') as f:
                                json.dump(job_data, f)
                            job_volume.commit()
                            last_commit_time = time.time()
                            
                    except json.JSONDecodeError:
                        # Skip non-JSON messages (binary previews)
                        continue
                    except websocket.WebSocketTimeoutException:
                        continue
                    except Exception as e:
                        print(f"WebSocket error: {e}")
                        break
                    
            finally:
                ws.close()
            
            # Get outputs with retry logic
            outputs = []
            for retry in range(3):
                try:
                    history_response = requests.get(
                        f"http://localhost:8188/history/{prompt_id}",
                        timeout=30
                    )
                    history = history_response.json()
                    
                    if prompt_id in history:
                        for node_id, node_output in history[prompt_id].get("outputs", {}).items():
                            # Handle text outputs
                            if "text" in node_output:
                                for text_item in node_output["text"]:
                                    outputs.append({
                                        "node_id": node_id,
                                        "type": "text",
                                        "data": text_item,
                                        "filename": f"text_output_{node_id}.txt",
                                        "size_bytes": len(text_item.encode('utf-8'))
                                    })
                            
                            # Handle UI/custom outputs (JSON data)
                            if "ui" in node_output:
                                ui_data = json.dumps(node_output["ui"])
                                outputs.append({
                                    "node_id": node_id,
                                    "type": "json",
                                    "data": ui_data,
                                    "filename": f"ui_output_{node_id}.json",
                                    "size_bytes": len(ui_data.encode('utf-8'))
                                })
                            
                            # Handle media outputs (existing code)
                            for media_type in ["images", "videos", "gifs", "audio"]:
                                if media_type in node_output:
                                    for file_info in node_output[media_type]:
                                        # Get file with timeout
                                        file_response = requests.get(
                                            "http://localhost:8188/view",
                                            params={
                                                "filename": file_info["filename"],
                                                "subfolder": file_info.get("subfolder", ""),
                                                "type": file_info.get("type", "output")
                                            },
                                            timeout=60  # Longer timeout for large files
                                        )
                                        
                                        outputs.append({
                                            "filename": file_info["filename"],
                                            "data": base64.b64encode(file_response.content).decode(),
                                            "type": self._get_media_type(file_info["filename"]),
                                            "size_bytes": len(file_response.content)
                                        })
                        break
                except Exception as e:
                    if retry == 2:
                        raise
                    print(f"Retry {retry + 1}: Failed to get outputs: {e}")
                    time.sleep(2)
            
            # Update job as completed
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            job_data["outputs"] = outputs
            job_data["progress"] = 100
            
            with open(job_file, 'w') as f:
                json.dump(job_data, f)
            job_volume.commit()
            
            # Send webhook
            if job_data.get("webhook_url"):
                self._send_webhook(job_data["webhook_url"], job_id, "job.completed")
            
            print(f"✨ Job {job_id} completed successfully!")
            
        except Exception as e:
            print(f"❌ Job {job_id} failed: {e}")
            # Update job as failed
            try:
                job_file = Path(JOB_DIR) / f"{job_id}.json"
                if job_file.exists():
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    
                    job_data["status"] = "failed"
                    job_data["error"] = str(e)
                    job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
                    
                    with open(job_file, 'w') as f:
                        json.dump(job_data, f)
                    job_volume.commit()
                    
                    if job_data.get("webhook_url"):
                        self._send_webhook(job_data["webhook_url"], job_id, "job.failed")
            except:
                pass
            
            raise
    
    def _get_media_type(self, filename: str) -> str:
        """Enhanced media type detection"""
        ext = Path(filename).suffix[1:].lower() if Path(filename).suffix else ""
        
        media_types = {
            "video": {"mp4", "webm", "avi", "mov", "mkv", "flv", "wmv", "m4v", "mpg", "mpeg"},
            "audio": {"wav", "mp3", "flac", "ogg", "aac", "m4a", "wma", "opus"},
            "image": {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "webp", "svg", "ico"},
            "document": {"pdf", "txt", "doc", "docx", "json", "xml", "csv"},
        }
        
        for media_type, extensions in media_types.items():
            if ext in extensions:
                return media_type
        return "file"
    
    def _send_webhook(self, webhook_url: str, job_id: str, event: str):
        """Send webhook with retry"""
        try:
            import httpx
            payload = {
                "event": event,
                "job_id": job_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            with httpx.Client(timeout=10) as client:
                client.post(webhook_url, json=payload)
        except:
            pass  # Silently fail

# FastAPI app
web_app = FastAPI(
    title="ComfyUI SaaS API",
    description="Production-ready ComfyUI API with job queues and webhooks",
    version="2.1.0"
)

@web_app.post("/v1/jobs", response_model=JobResponse)
async def create_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Submit a new ComfyUI job"""
    if not request.workflow:
        raise HTTPException(status_code=400, detail="Workflow cannot be empty")
    
    if len(request.media) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 media files allowed")
    
    for media in request.media:
        if media.data:
            estimated_size = len(media.data) * 3 / 4
            if estimated_size > 50 * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"Media file {media.name} exceeds 50MB limit")
    
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    
    job_data = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now.isoformat(),
        "workflow": request.workflow,
        "inputs": [inp.dict() for inp in request.inputs],
        "media": [media.dict() for media in request.media],
        "webhook_url": request.webhook_url,
        "priority": request.priority,
        "progress": 0,
        "outputs": []
    }
    
    # Save job
    job_file = Path(JOB_DIR) / f"{job_id}.json"
    job_file.parent.mkdir(parents=True, exist_ok=True)
    with open(job_file, 'w') as f:
        json.dump(job_data, f)
    job_volume.commit()
    
    # Submit job asynchronously (use .remote() not .spawn())
    ComfyService().process_job.remote(job_id)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=now,
        estimated_time=30
    )

@web_app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get job status with caching"""
    # Only reload if needed (check file modification time)
    job_file = Path(JOB_DIR) / f"{job_id}.json"
    
    # Implement simple caching to reduce volume reloads
    cache_key = f"{job_id}_cache"
    if not hasattr(get_job_status, cache_key):
        job_volume.reload()
    
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    with open(job_file, 'r') as f:
        job_data = json.load(f)
    
    # Cache for 2 seconds for running jobs, longer for completed
    if job_data["status"] in ["completed", "failed"]:
        setattr(get_job_status, cache_key, True)
    
    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        created_at=datetime.fromisoformat(job_data["created_at"]),
        started_at=datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None,
        completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
        progress=job_data.get("progress", 0),
        outputs=job_data.get("outputs", []),
        error=job_data.get("error")
    )

@web_app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a job"""
    job_volume.reload()
    job_file = Path(JOB_DIR) / f"{job_id}.json"
    
    if not job_file.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    
    with open(job_file, 'r') as f:
        job_data = json.load(f)
    
    if job_data["status"] in ["completed", "failed"]:
        return {"message": "Job already completed"}
    
    job_data["status"] = "failed"
    job_data["error"] = "Cancelled by user"
    job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
    
    with open(job_file, 'w') as f:
        json.dump(job_data, f)
    job_volume.commit()
    
    return {"message": "Job cancelled"}

@web_app.get("/v1/jobs")
async def list_jobs(status: Optional[JobStatus] = None, limit: int = 50):
    """List recent jobs"""
    job_volume.reload()
    job_path = Path(JOB_DIR)
    
    jobs = []
    if job_path.exists():
        for job_file in sorted(job_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:limit]:
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                if not status or job_data["status"] == status:
                    jobs.append({
                        "job_id": job_data["job_id"],
                        "status": job_data["status"],
                        "created_at": job_data["created_at"],
                        "progress": job_data.get("progress", 0)
                    })
            except:
                continue
    
    return {"jobs": jobs}

@web_app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

# Deploy the API
@app.function(
    image=image.pip_install("fastapi[standard]"),
    max_containers=3,
    volumes={JOB_DIR: job_volume}
)
@modal.asgi_app()
def api():
    return web_app

# Verify setup
@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
    timeout=300
)
def verify_setup():
    """Run: modal run your_script.py::verify_setup"""
    print("🔍 Verifying setup...")
    
    models_path = Path(f"{CACHE_DIR}/models")
    if models_path.exists():
        models = list(models_path.rglob("*.safetensors"))
        print(f"✅ Found {len(models)} model files")
        total_size = sum(m.stat().st_size for m in models) / (1024**3)
        print(f"📊 Total size: {total_size:.2f} GB")
    
    nodes_path = Path("/root/comfy/ComfyUI/custom_nodes")
    if nodes_path.exists():
        nodes = [d for d in nodes_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
        print(f"✅ Found {len(nodes)} custom nodes")
    
    print("\n🎉 Setup verified! Deploy with: modal deploy your_script.py")

# Cleanup
@app.function(schedule=modal.Cron("0 2 * * *"))
def cleanup_old_jobs():
    """Remove old jobs and temp files"""
    from datetime import timedelta
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    
    # Clean old jobs
    job_path = Path(JOB_DIR)
    if job_path.exists():
        for job_file in job_path.glob("*.json"):
            try:
                with open(job_file, 'r') as f:
                    job_data = json.load(f)
                created = datetime.fromisoformat(job_data["created_at"])
                if created < cutoff:
                    job_file.unlink()
                    print(f"Deleted old job: {job_file.name}")
            except:
                continue
        job_volume.commit()
    
    # Clean temp files
    temp_path = Path(f"{CACHE_DIR}/temp")
    if temp_path.exists():
        for temp_file in temp_path.iterdir():
            try:
                if temp_file.stat().st_mtime < cutoff.timestamp():
                    temp_file.unlink()
            except:
                continue
        cache_volume.commit()