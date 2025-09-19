import json
import uuid
import base64
import time
import subprocess
import threading
import shutil
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Any
from enum import Enum

import modal
import requests
import websocket
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field

APP_NAME = "comfyui-saas-api"
app = modal.App(APP_NAME)

# CRITICAL: Single cache volume mounted at EMPTY path
cache_volume = modal.Volume.from_name("comfyui-cache", create_if_missing=True)
CACHE_DIR = "/cache"

# Job storage volume (persistent across deployments)
job_volume = modal.Volume.from_name("job-storage", create_if_missing=True)
JOB_DIR = "/jobs"

# Guards volume reload/commit so concurrent threads in the same container
# (enabled via @modal.concurrent) do not reload while another one still has
# a file handle open. This avoids "there are open files" runtime errors.
job_volume_lock = RLock()

@contextmanager
def job_volume_guard(*, reload: bool = False, commit: bool = False):
    with job_volume_lock:
        if reload:
            job_volume.reload()
        try:
            yield
        except Exception:
            raise
        else:
            if commit:
                job_volume.commit()

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
    inputs: List[WorkflowInput] = Field(default_factory=list)
    media: List[MediaFile] = Field(default_factory=list)
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
    outputs: List[Dict] = Field(default_factory=list)
    error: Optional[str] = None
    prompt_id: Optional[str] = None
    nodes_total: int = 0
    nodes_done: int = 0
    current_node: Optional[str] = None
    last_event_time: Optional[datetime] = None
    error_log_tail: List[str] = Field(default_factory=list)

def download_assets_and_setup():
    """Download model files during image build"""
    from huggingface_hub import hf_hub_download
    import subprocess

    print("🚀 Setting up ComfyUI models...")

    # Create directory structure
    for subdir in ["models/clip_vision", "models/diffusion_models",
                   "models/text_encoders", "models/vae", "models/checkpoints",
                   "models/loras", "models/controlnet", "models/upscale_models",
                   "models/vae_approx", "outputs", "temp"]:
        Path(f"{CACHE_DIR}/{subdir}").mkdir(parents=True, exist_ok=True)
    
    # Download models using HF Transfer
    model_files = [
        # Existing Wan 2.1 models
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

        # New Wan 2.2 Lightning models
        ("Kijai/WanVideo_comfy",
         "Wan22-Lightning/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors",
         f"{CACHE_DIR}/models/loras"),
        ("Kijai/WanVideo_comfy",
         "Wan22-Lightning/Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors",
         f"{CACHE_DIR}/models/loras"),
        ("Kijai/WanVideo_comfy_fp8_scaled",
         "I2V/Wan2_2-I2V-A14B-HIGH_fp8_e5m2_scaled_KJ.safetensors",
         f"{CACHE_DIR}/models/diffusion_models"),
        ("Kijai/WanVideo_comfy_fp8_scaled",
         "I2V/Wan2_2-I2V-A14B-LOW_fp8_e5m2_scaled_KJ.safetensors",
         f"{CACHE_DIR}/models/diffusion_models"),
        ("Kijai/WanVideo_comfy",
         "Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
         f"{CACHE_DIR}/models/loras"),
        ("Kijai/WanVideo_comfy",
         "Wan22-Lightning/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
         f"{CACHE_DIR}/models/loras"),
        ("Kijai/WanVideo_comfy",
         "Wan2_1_VAE_fp32.safetensors",
         f"{CACHE_DIR}/models/vae"),
        ("Kijai/WanVideo_comfy",
         "umt5-xxl-enc-bf16.safetensors",
         f"{CACHE_DIR}/models/text_encoders"),

        # Additional clip vision
        ("Kijai/WanVideo_comfy",
         "open-clip-xlm-roberta-large-vit-huge-14_visual_fp32.safetensors",
         f"{CACHE_DIR}/models/clip_vision"),

        # VAE Approx and Upscaler
        ("Kijai/WanVideo_comfy",
         "taew2_1.safetensors",
         f"{CACHE_DIR}/models/vae_approx"),
        ("ABDALLALSWAITI/Upscalers",
         "anime/2x-AnimeSharpV2_MoSR_Soft.pth",
         f"{CACHE_DIR}/models/upscale_models"),
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

    # Download Mega LoRAs
    print("📥 Downloading LoRAs from Mega...")
    mega_loras = [
        "https://mega.nz/file/xJ4w3DiT#E5h8kIwr-PQuxG4EeHXRw4uL1VFEbLxjYCMofsdfxNI",
        "https://mega.nz/file/8NhkGTAA#Ww8jci3_uL9c7YzKJNw2MqX5vBUDc3ANJzUiaOGvHkk",
    ]

    for mega_url in mega_loras:
        try:
            print(f"📥 Downloading from Mega: {mega_url}")

            # Get list of files before download to track what's new
            import os
            before_files = set(os.listdir(f"{CACHE_DIR}/models/loras/")) if os.path.exists(f"{CACHE_DIR}/models/loras/") else set()

            # Use megadl to download the file (it will use original name)
            result = subprocess.run(
                ["megadl", mega_url, "--path", f"{CACHE_DIR}/models/loras/", "--no-progress"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                # Find what file was downloaded
                after_files = set(os.listdir(f"{CACHE_DIR}/models/loras/"))
                new_files = after_files - before_files

                if new_files:
                    downloaded_file = new_files.pop()
                    print(f"✅ Downloaded: {downloaded_file}")

                    # Check if file already existed with same size
                    file_path = Path(f"{CACHE_DIR}/models/loras/{downloaded_file}")
                    if file_path.exists() and file_path.stat().st_size > 1000000:
                        print(f"✅ File {downloaded_file} downloaded successfully ({file_path.stat().st_size / (1024*1024):.2f} MB)")
                else:
                    print(f"✅ File already exists (megadl skipped download)")
            else:
                print(f"⚠️ Failed to download from Mega: {result.stderr}")
        except Exception as e:
            print(f"⚠️ Failed to download from Mega: {e}")
            continue

    cache_volume.commit()
    print("✅ Models committed to volume!")

def install_custom_nodes():
    """Install custom nodes with proper error handling"""
    import subprocess
    
    nodes_dir = Path("/root/comfy/ComfyUI/custom_nodes")
    
    custom_nodes = [
        # Existing nodes
        "https://github.com/pythongosssss/ComfyUI-Custom-Scripts",
        "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
        "https://github.com/ltdrdata/ComfyUI-Manager",
        "https://github.com/WASasquatch/was-node-suite-comfyui",
        "https://github.com/cubiq/ComfyUI_essentials",
        "https://github.com/kijai/ComfyUI-KJNodes",

        # New nodes
        "https://github.com/kijai/ComfyUI-Florence2",
        "https://github.com/kijai/ComfyUI-WanVideoWrapper",
        "https://github.com/kijai/ComfyUI-GIMM-VFI",
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "https://github.com/Fannovel16/comfyui_controlnet_aux",
        "https://github.com/Artificial-Sweetener/comfyui-WhiteRabbit",
        "https://github.com/shiimizu/ComfyUI_smZNodes",
        "https://github.com/CoreyCorza/ComfyUI-CRZnodes",
        "https://github.com/yuvraj108c/ComfyUI-Dwpose-Tensorrt",
        "https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes",
        "https://github.com/grmchn/ComfyUI-ProportionChanger",
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
    .apt_install("git", "wget", "curl", "libgl1", "libglib2.0-0", "ffmpeg", "megatools")
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
        "slowapi",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PYTHONUNBUFFERED": "1",  # Better for logging
    })
    # Install ComfyUI
    .run_commands(
        "cd /root && comfy --skip-prompt install --skip-manager --fast-deps --nvidia",
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

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]",
        "slowapi",
        "pydantic",
        "requests",
        "websocket-client",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
    })
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
    max_containers=4,
)
@modal.concurrent(max_inputs=1)
class ComfyService:
    
    @modal.enter(snap=True)  # Snapshot WITHOUT starting ComfyUI
    def setup_environment(self):
        """Setup environment and preload libraries - GPU not available during snapshot"""
        import sys
        import os
        
        print("📸 Creating memory snapshot (no GPU available yet)...")
        
        # Pre-import CPU-only libraries so the snapshot stays GPU-agnostic
        import numpy  # Installed as torch dependency
        import PIL  # From pillow package
        import cv2  # From opencv-python-headless
        
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
        
        # Initialize runtime state containers (populated when GPU available)
        self.process = None
        self._process_watch_thread: Optional[threading.Thread] = None
        self._log_tail = deque(maxlen=200)
    
    @modal.enter(snap=False)  # Run AFTER snapshot restore, when GPU is available
    def start_comfy_with_gpu(self):
        """Start ComfyUI server with retry logic when GPU is available"""
        import subprocess
        import time

        print("🎮 GPU now available, starting ComfyUI server...")

        if self.process and self.process.poll() is None:
            print("♻️ ComfyUI server already running in this container.")
            return

        max_retries = 3
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retries + 1):
            try:
                print(f"📡 Starting ComfyUI server (attempt {attempt}/{max_retries})...")

                self._terminate_process()

                self.process = subprocess.Popen(
                    [
                        "python",
                        "/root/comfy/ComfyUI/main.py",
                        "--listen",
                        "0.0.0.0",
                        "--port",
                        "8188",
                        "--use-sage-attention",
                        "--disable-auto-launch",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

                def _watch_logs():
                    try:
                        for line in iter(self.process.stdout.readline, ""):
                            if not line:
                                break
                            print(line, end="")
                            self._log_tail.append(line.strip())
                            if "Traceback" in line:
                                print("⚠️ ComfyUI emitted a traceback; continuing to monitor.")
                    finally:
                        try:
                            if self.process.stdout:
                                self.process.stdout.close()
                        except Exception:
                            pass

                self._process_watch_thread = threading.Thread(target=_watch_logs, daemon=True)
                self._process_watch_thread.start()

                backoff = 2.0
                deadline = time.time() + 180
                with requests.Session() as session:
                    while time.time() < deadline:
                        if self.process.poll() is not None:
                            raise RuntimeError("ComfyUI process exited prematurely")
                        try:
                            resp = session.get("http://localhost:8188/system_stats", timeout=3)
                            if resp.status_code == 200:
                                print("✅ ComfyUI server ready with GPU!")
                                break
                        except Exception:
                            pass
                        time.sleep(backoff)
                        backoff = min(backoff * 1.5, 10.0)
                    else:
                        raise TimeoutError("ComfyUI server did not become ready within 180 seconds")

                import torch

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available after ComfyUI start")

                print(f"✅ CUDA initialized. Devices: {torch.cuda.device_count()}")
                return

            except Exception as exc:
                last_error = exc
                print(f"⚠️ ComfyUI start attempt {attempt} failed: {exc}")
                self._terminate_process()
                if attempt < max_retries:
                    time.sleep(min(5 * attempt, 15))

        raise RuntimeError(f"Failed to start ComfyUI after {max_retries} attempts: {last_error}")

    def _terminate_process(self):
        """Terminate the managed ComfyUI process if it is running."""
        if not self.process:
            return
        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
        finally:
            self.process = None
        if getattr(self, "_process_watch_thread", None):
            self._process_watch_thread.join(timeout=2)
            self._process_watch_thread = None

    def _persist_job(self, job_file: Path, job_data: Dict[str, Any], *, commit: bool, touch_event_time: bool = True) -> None:
        """Persist job metadata to the shared volume."""
        if touch_event_time:
            job_data["last_event_time"] = datetime.now(timezone.utc).isoformat()
        job_file.parent.mkdir(parents=True, exist_ok=True)
        with job_volume_guard(commit=commit):
            with open(job_file, "w") as f:
                json.dump(job_data, f)

    def _request_with_retry(self, method: str, url: str, *, retries: int = 3, backoff: float = 1.5, **kwargs) -> requests.Response:
        """Perform an HTTP request with simple exponential backoff."""
        delay = 1.0
        for attempt in range(retries):
            try:
                response = requests.request(method=method, url=url, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay *= backoff

    @modal.method()
    def process_job(self, job_id: str):
        """Process a ComfyUI job"""

        job_file = Path(JOB_DIR) / f"{job_id}.json"
        job_input_dir = INPUT_DIR / job_id

        try:
            with job_volume_guard(reload=True):
                if not job_file.exists():
                    raise FileNotFoundError(f"Job {job_id} not found on volume")
                with open(job_file, "r") as f:
                    job_data = json.load(f)

            now_iso = datetime.now(timezone.utc).isoformat()
            job_data["status"] = "running"
            job_data["started_at"] = now_iso
            job_data.setdefault("progress", 0)
            job_data.setdefault("outputs", [])
            job_data["prompt_id"] = None
            job_data["nodes_total"] = 0
            job_data["nodes_done"] = 0
            job_data["current_node"] = None
            job_data["last_event_time"] = now_iso

            self._persist_job(job_file, job_data, commit=True, touch_event_time=False)

            workflow = json.loads(json.dumps(job_data["workflow"]))
            inputs = job_data.get("inputs", [])
            media = job_data.get("media", [])

            if job_input_dir.exists():
                shutil.rmtree(job_input_dir, ignore_errors=True)
            job_input_dir.mkdir(parents=True, exist_ok=True)

            media_remap: Dict[str, str] = {}

            def _store_media_bytes(filename: str, payload: bytes) -> str:
                safe_name = Path(filename).name or f"asset_{uuid.uuid4().hex}"
                dest = job_input_dir / safe_name
                dest.write_bytes(payload)
                rel_path = str(Path(job_id) / safe_name)
                media_remap[safe_name] = rel_path
                return rel_path

            for item in media:
                name = item.get("name") or f"media_{uuid.uuid4().hex}"
                try:
                    if item.get("data"):
                        _store_media_bytes(name, base64.b64decode(item["data"]))
                    elif item.get("url"):
                        resp = self._request_with_retry("get", item["url"], timeout=30)
                        content_type = resp.headers.get("Content-Type", "")
                        if content_type.startswith("text/") and "json" not in content_type:
                            resp.close()
                            raise ValueError(f"Unexpected content type '{content_type}' for media '{name}'")
                        _store_media_bytes(name, resp.content)
                        resp.close()
                except Exception as exc:
                    raise RuntimeError(f"Failed to prepare media '{name}': {exc}") from exc

            def _apply_media_remap(obj: Any) -> Any:
                if isinstance(obj, dict):
                    return {k: _apply_media_remap(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_apply_media_remap(v) for v in obj]
                if isinstance(obj, str) and obj in media_remap:
                    return media_remap[obj]
                return obj

            if media_remap:
                workflow = _apply_media_remap(workflow)

            for inp in inputs:
                node_id = str(inp.get("node"))
                if node_id not in workflow:
                    continue
                workflow.setdefault(node_id, {}).setdefault("inputs", {})
                field = inp.get("field")
                value = inp.get("value")
                input_type = inp.get("type", "raw")

                try:
                    if input_type == "image_base64" and isinstance(value, str):
                        filename = f"input_{node_id}_{field}.png"
                        rel_path = _store_media_bytes(filename, base64.b64decode(value))
                        workflow[node_id]["inputs"][field] = rel_path
                    elif input_type == "image_url" and isinstance(value, str):
                        resp = self._request_with_retry("get", value, timeout=30)
                        content_type = resp.headers.get("Content-Type", "")
                        if not content_type.startswith("image/"):
                            resp.close()
                            raise ValueError(f"URL for node {node_id} returned unsupported content type '{content_type}'")
                        filename = f"input_{node_id}_{field}.png"
                        rel_path = _store_media_bytes(filename, resp.content)
                        resp.close()
                        workflow[node_id]["inputs"][field] = rel_path
                    else:
                        workflow[node_id]["inputs"][field] = value
                except Exception as exc:
                    raise RuntimeError(f"Failed to prepare dynamic input for node {node_id}") from exc

            try:
                health_resp = requests.get("http://127.0.0.1:8188/system_stats", timeout=3)
                health_resp.raise_for_status()
            except Exception as exc:
                raise RuntimeError("ComfyUI not healthy in this container") from exc

            client_id = str(uuid.uuid4())
            ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
            ws = None

            prompt_id: Optional[str] = None
            executed_nodes: set[str] = set()
            cached_nodes: set[str] = set()
            node_order: List[str] = []
            node_lookup: Dict[str, int] = {}
            current_partial = [0.0, 1.0]
            last_commit_ts = time.time()
            last_commit_progress = job_data.get("progress", 0)
            last_history_poll = 0.0

            def _set_node_order(nodes: List[Any]) -> None:
                cleaned: List[str] = []
                for node in nodes:
                    if node is None:
                        continue
                    node_str = str(node)
                    if node_str not in cleaned:
                        cleaned.append(node_str)
                if not cleaned:
                    return
                node_order.clear()
                node_order.extend(cleaned)
                node_lookup.clear()
                node_lookup.update({nid: idx for idx, nid in enumerate(node_order)})

            def _compute_progress() -> int:
                total_planned = len(node_order) if node_order else len(workflow)
                total_effective = max(1, total_planned - len(cached_nodes))
                done = min(total_effective, len(executed_nodes))
                fraction = done / total_effective
                if current_partial[1] and current_partial[1] > 0 and done < total_effective:
                    fraction += (current_partial[0] / current_partial[1]) / total_effective
                fraction = max(0.0, min(1.0, fraction))
                return int(round(fraction * 100))

            def _persist(progress_changed: bool = False, force_commit: bool = False) -> None:
                nonlocal last_commit_ts, last_commit_progress
                now_ts = time.time()
                commit = force_commit or job_data.get("progress", 0) >= 100
                if progress_changed and abs(job_data.get("progress", 0) - last_commit_progress) >= 5:
                    commit = True
                if now_ts - last_commit_ts >= 30:
                    commit = True
                self._persist_job(job_file, job_data, commit=commit)
                if commit:
                    last_commit_ts = now_ts
                    last_commit_progress = job_data.get("progress", 0)

            try:
                ws = websocket.create_connection(ws_url, timeout=15)
                ws.settimeout(10)

                submit_payload = {
                    "prompt": workflow,
                    "client_id": client_id,
                    "extra_data": {"extra_pnginfo": {"workflow": workflow}},
                }
                response = self._request_with_retry(
                    "post",
                    "http://127.0.0.1:8188/prompt",
                    json=submit_payload,
                    timeout=60,
                )
                prompt_id = response.json()["prompt_id"]
                print(f"📋 Executing prompt {prompt_id}")

                job_data["prompt_id"] = prompt_id
                _set_node_order(list(workflow.keys()))
                job_data["nodes_total"] = len(node_order)
                job_data["progress"] = 0
                _persist(force_commit=True)

                while True:
                    try:
                        raw_msg = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        now_ts = time.time()
                        if prompt_id and now_ts - last_history_poll >= 5:
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
                                workflow_nodes = (
                                    prompt_history.get("workflow", {}).get("nodes", [])
                                )
                                if workflow_nodes and not node_order:
                                    _set_node_order([n.get("id") for n in workflow_nodes if n.get("id")])
                                    job_data["nodes_total"] = len(node_order)
                                outputs = prompt_history.get("outputs", {})
                                if outputs:
                                    executed_nodes.update(str(node) for node in outputs.keys())
                                    job_data["nodes_done"] = len(executed_nodes)
                                    prev = job_data.get("progress", 0)
                                    job_data["progress"] = _compute_progress()
                                    _persist(progress_changed=job_data["progress"] != prev)
                            except Exception as poll_err:
                                print(f"Progress poll failed: {poll_err}")
                        continue

                    if isinstance(raw_msg, bytes):
                        continue

                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        continue

                    msg_type = msg.get("type")
                    data = msg.get("data", {})

                    if msg_type == "execution_start" and data.get("prompt_id") == prompt_id:
                        nodes = data.get("nodes") or []
                        _set_node_order(nodes)
                        job_data["nodes_total"] = len(node_order)
                        _persist()
                        continue

                    if msg_type == "executing" and data.get("prompt_id") == prompt_id:
                        node = data.get("node")
                        if node is None:
                            print("✅ Execution complete!")
                            break
                        job_data["current_node"] = str(node)
                        _persist()
                        continue

                    if msg_type == "execution_cached" and data.get("prompt_id") == prompt_id:
                        node = data.get("node")
                        if node is not None:
                            cached_nodes.add(str(node))
                            executed_nodes.add(str(node))
                            job_data["nodes_done"] = len(executed_nodes)
                            prev = job_data.get("progress", 0)
                            job_data["progress"] = _compute_progress()
                            _persist(progress_changed=job_data["progress"] != prev)
                        continue

                    if msg_type == "executed" and data.get("prompt_id") == prompt_id:
                        node = data.get("node")
                        if node is not None:
                            executed_nodes.add(str(node))
                            job_data["nodes_done"] = len(executed_nodes)
                            job_data["current_node"] = None
                            prev = job_data.get("progress", 0)
                            job_data["progress"] = _compute_progress()
                            _persist(progress_changed=job_data["progress"] != prev)
                        continue

                    if msg_type == "progress" and data.get("prompt_id") == prompt_id:
                        current_partial[0] = float(data.get("value", 0) or 0)
                        current_partial[1] = float(data.get("max", 1) or 1)
                        prev = job_data.get("progress", 0)
                        job_data["progress"] = _compute_progress()
                        _persist(progress_changed=job_data["progress"] != prev)
                        continue

                    if msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                        raise RuntimeError(f"Execution error: {data}")

            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:
                        pass

            outputs: List[Dict[str, Any]] = []
            if prompt_id:
                for retry in range(3):
                    try:
                        history_response = self._request_with_retry(
                            "get",
                            f"http://127.0.0.1:8188/history/{prompt_id}",
                            timeout=30,
                        )
                        history = history_response.json()
                        history_response.close()

                        if prompt_id in history:
                            for node_id, node_output in history[prompt_id].get("outputs", {}).items():
                                if "text" in node_output:
                                    for text_item in node_output["text"]:
                                        outputs.append({
                                            "node_id": node_id,
                                            "type": "text",
                                            "data": text_item,
                                            "filename": f"text_output_{node_id}.txt",
                                            "size_bytes": len(text_item.encode("utf-8")),
                                        })
                                if "ui" in node_output:
                                    ui_data = json.dumps(node_output["ui"])
                                    outputs.append({
                                        "node_id": node_id,
                                        "type": "json",
                                        "data": ui_data,
                                        "filename": f"ui_output_{node_id}.json",
                                        "size_bytes": len(ui_data.encode("utf-8")),
                                    })
                                for media_type in ["images", "videos", "gifs", "audio"]:
                                    if media_type in node_output:
                                        for file_info in node_output[media_type]:
                                            with self._request_with_retry(
                                                "get",
                                                "http://127.0.0.1:8188/view",
                                                params={
                                                    "filename": file_info["filename"],
                                                    "subfolder": file_info.get("subfolder", ""),
                                                    "type": file_info.get("type", "output"),
                                                },
                                                timeout=120,
                                                stream=True,
                                            ) as file_response:
                                                chunks: List[bytes] = []
                                                for chunk in file_response.iter_content(1 << 20):
                                                    if chunk:
                                                        chunks.append(chunk)
                                                payload = b"".join(chunks)
                                            outputs.append({
                                                "filename": file_info["filename"],
                                                "data": base64.b64encode(payload).decode(),
                                                "type": self._get_media_type(file_info["filename"]),
                                                "size_bytes": len(payload),
                                            })
                            break
                    except Exception as e:
                        if retry == 2:
                            raise
                        print(f"Retry {retry + 1}: Failed to get outputs: {e}")
                        time.sleep(2)

            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            job_data["outputs"] = outputs
            job_data["progress"] = 100
            job_data["current_node"] = None
            job_data["nodes_done"] = len(executed_nodes) if executed_nodes else job_data.get("nodes_done", 0)

            self._persist_job(job_file, job_data, commit=True)

            if job_data.get("webhook_url"):
                self._send_webhook(job_data["webhook_url"], job_id, "job.completed")

            print(f"✨ Job {job_id} completed successfully!")

        except Exception as e:
            print(f"❌ Job {job_id} failed: {e}")
            try:
                if job_file.exists():
                    with job_volume_guard(reload=True):
                        with open(job_file, "r") as f:
                            job_data = json.load(f)
                else:
                    job_data = {
                        "job_id": job_id,
                        "status": "failed",
                    }

                job_data["status"] = "failed"
                job_data["error"] = str(e)
                job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
                job_data.setdefault("error_log_tail", list(self._log_tail)[-50:])

                self._persist_job(job_file, job_data, commit=True)

                if job_data.get("webhook_url"):
                    self._send_webhook(job_data["webhook_url"], job_id, "job.failed")
            except Exception as inner_err:
                print(f"⚠️ Failed to persist failure state for job {job_id}: {inner_err}")
            raise
        finally:
            shutil.rmtree(job_input_dir, ignore_errors=True)

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
            backoff = 1.0
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=10) as client:
                        response = client.post(webhook_url, json=payload)
                        response.raise_for_status()
                    return
                except Exception as exc:
                    if attempt == 2:
                        print(f"⚠️ Webhook delivery failed for job {job_id}: {exc}")
                    time.sleep(backoff)
                    backoff *= 2
        except Exception as outer_exc:
            print(f"⚠️ Webhook exception for job {job_id}: {outer_exc}")

# FastAPI app
web_app = FastAPI(
    title="ComfyUI SaaS API",
    description="Production-ready ComfyUI API with job queues and webhooks",
    version="2.1.0"
)

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
web_app.state.limiter = limiter
web_app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
web_app.add_middleware(SlowAPIMiddleware)

@web_app.post("/v1/jobs", response_model=JobResponse)
@limiter.limit("10/minute")
async def create_job(request: Request, job_request: JobRequest):
    """Submit a new ComfyUI job"""
    if not job_request.workflow:
        raise HTTPException(status_code=400, detail="Workflow cannot be empty")
    
    if len(job_request.media) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 media files allowed")
    
    for media in job_request.media:
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
        "last_event_time": now.isoformat(),
        "workflow": job_request.workflow,
        "inputs": [inp.model_dump() for inp in job_request.inputs],
        "media": [media.model_dump() for media in job_request.media],
        "webhook_url": job_request.webhook_url,
        "priority": job_request.priority,
        "progress": 0,
        "outputs": [],
        "prompt_id": None,
        "nodes_total": 0,
        "nodes_done": 0,
        "current_node": None,
        "error_log_tail": []
    }
    
    # Save job
    job_file = Path(JOB_DIR) / f"{job_id}.json"
    with job_volume_guard(commit=True):
        job_file.parent.mkdir(parents=True, exist_ok=True)
        with open(job_file, 'w') as f:
            json.dump(job_data, f)
    
    # Submit job asynchronously; spawn() schedules the work without blocking
    ComfyService().process_job.spawn(job_id)
    
    return JobResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=now,
        estimated_time=30
    )

@web_app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
@limiter.limit("120/minute")
async def get_job_status(request: Request, job_id: str):
    """Get job status"""
    job_file = Path(JOB_DIR) / f"{job_id}.json"

    with job_volume_guard(reload=True):
        if not job_file.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        with open(job_file, 'r') as f:
            job_data = json.load(f)

    return JobStatusResponse(
        job_id=job_id,
        status=job_data["status"],
        created_at=datetime.fromisoformat(job_data["created_at"]),
        started_at=datetime.fromisoformat(job_data["started_at"]) if job_data.get("started_at") else None,
        completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
        progress=job_data.get("progress", 0),
        outputs=job_data.get("outputs", []),
        error=job_data.get("error"),
        prompt_id=job_data.get("prompt_id"),
        nodes_total=job_data.get("nodes_total", 0),
        nodes_done=job_data.get("nodes_done", 0),
        current_node=job_data.get("current_node"),
        last_event_time=datetime.fromisoformat(job_data["last_event_time"]) if job_data.get("last_event_time") else None,
        error_log_tail=job_data.get("error_log_tail", []),
    )

@web_app.delete("/v1/jobs/{job_id}")
@limiter.limit("30/minute")
async def cancel_job(request: Request, job_id: str):
    """Cancel a job"""
    job_file = Path(JOB_DIR) / f"{job_id}.json"

    with job_volume_guard(reload=True):
        if not job_file.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        with open(job_file, 'r') as f:
            job_data = json.load(f)

    if job_data["status"] in ["completed", "failed"]:
        return {"message": "Job already completed"}

    prompt_id = job_data.get("prompt_id")
    if job_data["status"] == "running" and prompt_id:
        try:
            requests.post("http://127.0.0.1:8188/interrupt", timeout=5)
            requests.post(
                "http://127.0.0.1:8188/queue/cancel",
                json={"prompt_id": prompt_id},
                timeout=5,
            )
        except Exception as exc:
            print(f"⚠️ Failed to cancel prompt {prompt_id}: {exc}")

    job_data["status"] = "failed"
    job_data["error"] = "Cancelled by user"
    job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
    job_data["current_node"] = None
    job_data["last_event_time"] = datetime.now(timezone.utc).isoformat()

    with job_volume_guard(commit=True):
        with open(job_file, 'w') as f:
            json.dump(job_data, f)

    return {"message": "Job cancelled"}

@web_app.get("/v1/jobs")
@limiter.limit("30/minute")
async def list_jobs(request: Request, status: Optional[JobStatus] = None, limit: int = 50):
    """List recent jobs"""
    job_path = Path(JOB_DIR)

    jobs = []
    with job_volume_guard(reload=True):
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
                except Exception:
                    continue
    
    return {"jobs": jobs}

@web_app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc)}

# Deploy the API
@app.function(
    image=api_image,
    max_containers=3,
    volumes={JOB_DIR: job_volume},
    timeout=300  # Increase timeout to 5 minutes for API endpoints
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
@app.function(
    schedule=modal.Cron("0 2 * * *"),
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={
        JOB_DIR: job_volume,
        CACHE_DIR: cache_volume
    }
)
def cleanup_old_jobs():
    """Remove old jobs and temp files"""
    import json
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    
    # Clean old jobs
    JOB_DIR = "/jobs"  # Define locally to avoid import issues
    job_path = Path(JOB_DIR)
    if job_path.exists():
        with job_volume_guard(reload=True, commit=True):
            for job_file in job_path.glob("*.json"):
                try:
                    with open(job_file, 'r') as f:
                        job_data = json.load(f)
                    created = datetime.fromisoformat(job_data["created_at"])
                    if created < cutoff:
                        job_file.unlink()
                        print(f"Deleted old job: {job_file.name}")
                except Exception:
                    continue
    
    # Clean temp files
    CACHE_DIR = "/cache"  # Define locally to avoid import issues
    temp_path = Path(f"{CACHE_DIR}/temp")
    if temp_path.exists():
        for temp_file in temp_path.iterdir():
            try:
                if temp_file.stat().st_mtime < cutoff.timestamp():
                    temp_file.unlink()
            except:
                continue
        cache_volume.commit()
