"""
ComfyUI SaaS API on Modal.com

Production-ready REST API that executes any ComfyUI workflow on GPU.
Outputs are uploaded to Cloudflare R2 and returned as signed URLs.

Deploy:  modal deploy comfyui_api.py
Test:    python test_api.py
"""

import json
import uuid
import base64
import time
import subprocess
import hashlib
import hmac
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from enum import Enum

import modal
import modal.experimental

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "comfyui-saas"
COMFY_PORT = 8188

# Volume names (persistent storage)
CACHE_VOL_NAME = "comfyui-models-cache"
JOBS_VOL_NAME = "comfyui-jobs"

# Paths inside the container
CACHE_DIR = "/cache"
JOBS_DIR = "/jobs"
COMFY_DIR = "/root/comfy/ComfyUI"

# Models to pre-download. Add entries here to include more models.
# Format: (hf_repo_id, hf_filename, target_subdir_under_CACHE_DIR/models)
MODELS = [
    # SDXL checkpoint for testing
    (
        "ChenkinNoob/ChenkinNoob-XL-V0.2",
        "ChenkinNoob-XL-V0.2.safetensors",
        "checkpoints",
    ),
]

# Custom nodes to install (add URLs here when workflows require them)
CUSTOM_NODES: list = []

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

cache_vol = modal.Volume.from_name(CACHE_VOL_NAME, create_if_missing=True)
jobs_vol = modal.Volume.from_name(JOBS_VOL_NAME, create_if_missing=True)

# Secrets for API auth and R2 storage
api_secret = modal.Secret.from_name("comfyui-api-secret", required_keys=["API_KEY"])
r2_secret = modal.Secret.from_name(
    "comfyui-r2",
    required_keys=[
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_ENDPOINT_URL",
        "R2_BUCKET_NAME",
    ],
)

# ---------------------------------------------------------------------------
# Image build helpers
# ---------------------------------------------------------------------------


def download_models():
    """Download model files to the cache volume during image build."""
    from huggingface_hub import hf_hub_download

    for repo_id, filename, subdir in MODELS:
        target_dir = Path(f"{CACHE_DIR}/models/{subdir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / Path(filename).name

        if target_path.exists() and target_path.stat().st_size > 1_000_000:
            print(f"[models] {filename} already cached, skipping")
            continue

        print(f"[models] Downloading {repo_id}/{filename} ...")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[models] OK {filename}")

    # Ensure output/temp dirs exist
    for d in ["outputs", "temp"]:
        Path(f"{CACHE_DIR}/{d}").mkdir(parents=True, exist_ok=True)

    cache_vol.commit()
    print("[models] All models committed to volume")


def install_custom_nodes():
    """Clone custom nodes into ComfyUI during image build."""
    if not CUSTOM_NODES:
        print("[nodes] No custom nodes configured, skipping")
        return

    nodes_dir = Path(f"{COMFY_DIR}/custom_nodes")

    for url in CUSTOM_NODES:
        name = url.rstrip("/").split("/")[-1]
        dest = nodes_dir / name
        if dest.exists():
            print(f"[nodes] {name} already installed")
            continue

        print(f"[nodes] Installing {name} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            timeout=120,
        )

        req = dest / "requirements.txt"
        if req.exists():
            subprocess.run(
                ["pip", "install", "-q", "-r", str(req)],
                check=False,
                timeout=120,
            )
        print(f"[nodes] OK {name}")


# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "curl", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "comfy-cli==1.4.1",
        "huggingface_hub[hf_transfer]==0.34.4",
        "websocket-client",
        "requests",
        "httpx",
        "boto3",
        "fastapi[standard]",
        "pydantic",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .run_commands(
        "comfy --skip-prompt install --skip-manager --fast-deps --nvidia",
    )
    .run_function(install_custom_nodes)
    .run_commands(
        # Symlink models and outputs to the cache volume mount point
        f"rm -rf {COMFY_DIR}/models",
        f"rm -rf {COMFY_DIR}/output",
        f"ln -s {CACHE_DIR}/models {COMFY_DIR}/models",
        f"ln -s {CACHE_DIR}/outputs {COMFY_DIR}/output",
        f"mkdir -p {COMFY_DIR}/input {COMFY_DIR}/temp",
    )
    .run_function(download_models, volumes={CACHE_DIR: cache_vol}, timeout=3600)
)

api_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "fastapi[standard]",
        "pydantic",
        "boto3",
    )
    .env({"PYTHONUNBUFFERED": "1"})
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field  # noqa: E402


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MediaInput(BaseModel):
    name: str
    data: Optional[str] = None  # base64
    url: Optional[str] = None


class WorkflowOverride(BaseModel):
    node: str
    field: str
    value: Any
    type: str = "raw"  # raw | image_base64 | image_url


class JobCreate(BaseModel):
    workflow: Dict
    inputs: List[WorkflowOverride] = Field(default_factory=list)
    media: List[MediaInput] = Field(default_factory=list)
    webhook_url: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: int = 0
    outputs: List[Dict] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# R2 helper (used inside the GPU worker)
# ---------------------------------------------------------------------------


def upload_to_r2(file_bytes: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to R2 and return the object key."""
    import os
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET_NAME"]
    s3.put_object(Bucket=bucket, Key=key, Body=file_bytes, ContentType=content_type)
    return key


def generate_r2_url(key: str, expires_in: int = 3600) -> str:
    """Generate a presigned URL for an R2 object."""
    import os
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET_NAME"]
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


# ---------------------------------------------------------------------------
# GPU Worker: ComfyService
# ---------------------------------------------------------------------------


@app.cls(
    gpu="L40S",
    image=gpu_image,
    volumes={CACHE_DIR: cache_vol, JOBS_DIR: jobs_vol},
    secrets=[r2_secret],
    enable_memory_snapshot=True,
    scaledown_window=120,
    max_containers=10,
    timeout=600,
)
@modal.concurrent(max_inputs=1)
class ComfyService:
    """Manages a ComfyUI server and processes workflow jobs."""

    @modal.enter(snap=True)
    def snapshot_phase(self):
        """Pre-load CPU libraries before snapshot (no GPU available)."""
        import numpy  # noqa: F401
        import PIL  # noqa: F401

        self.server_process = None
        self._healthy = True
        print("[snapshot] Environment ready for snapshot")

    @modal.enter(snap=False)
    def gpu_phase(self):
        """Start ComfyUI server when GPU becomes available."""
        import requests as req

        print("[gpu] Starting ComfyUI server ...")

        self.server_process = subprocess.Popen(
            [
                "python",
                f"{COMFY_DIR}/main.py",
                "--listen", "0.0.0.0",
                "--port", str(COMFY_PORT),
                "--disable-auto-launch",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Wait for server to become ready
        deadline = time.time() + 120
        delay = 1.0
        while time.time() < deadline:
            if self.server_process.poll() is not None:
                raise RuntimeError("ComfyUI process exited during startup")
            try:
                r = req.get(f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=3)
                if r.status_code == 200:
                    print("[gpu] ComfyUI server ready!")
                    return
            except Exception:
                pass
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)

        raise TimeoutError("ComfyUI did not start within 120s")

    def _check_health(self):
        """Check if ComfyUI is responsive; stop accepting work if not."""
        import requests as req

        try:
            r = req.get(f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=5)
            r.raise_for_status()
        except Exception as e:
            print(f"[health] ComfyUI unhealthy: {e}")
            self._healthy = False
            modal.experimental.stop_fetching_inputs()
            raise RuntimeError("ComfyUI server is not healthy, stopping container")

    @modal.method()
    def run_job(self, job_id: str):
        """Execute a queued job end-to-end."""
        import requests as req
        import websocket
        import shutil

        self._check_health()

        job_file = Path(JOBS_DIR) / f"{job_id}.json"
        input_dir = Path(f"{COMFY_DIR}/input") / job_id

        try:
            # 1. Load job data
            jobs_vol.reload()
            if not job_file.exists():
                raise FileNotFoundError(f"Job {job_id} not found")
            job_data = json.loads(job_file.read_text())

            # 2. Mark as running
            job_data["status"] = "running"
            job_data["started_at"] = datetime.now(timezone.utc).isoformat()
            self._save_job(job_file, job_data)

            workflow = json.loads(json.dumps(job_data["workflow"]))

            # 3. Prepare media files
            input_dir.mkdir(parents=True, exist_ok=True)
            media_remap = {}

            for item in job_data.get("media", []):
                name = item.get("name", f"media_{uuid.uuid4().hex[:8]}")
                safe_name = Path(name).name
                dest = input_dir / safe_name

                if item.get("data"):
                    dest.write_bytes(base64.b64decode(item["data"]))
                elif item.get("url"):
                    r = req.get(item["url"], timeout=30)
                    r.raise_for_status()
                    dest.write_bytes(r.content)

                media_remap[safe_name] = str(Path(job_id) / safe_name)

            # Remap media references in workflow
            if media_remap:
                workflow = self._remap(workflow, media_remap)

            # 4. Apply input overrides
            for inp in job_data.get("inputs", []):
                node_id = str(inp["node"])
                if node_id not in workflow:
                    continue
                workflow.setdefault(node_id, {}).setdefault("inputs", {})
                workflow[node_id]["inputs"][inp["field"]] = inp["value"]

            # 5. Submit to ComfyUI via WebSocket
            client_id = uuid.uuid4().hex
            ws = websocket.create_connection(
                f"ws://127.0.0.1:{COMFY_PORT}/ws?clientId={client_id}",
                timeout=15,
            )
            ws.settimeout(10)

            resp = req.post(
                f"http://127.0.0.1:{COMFY_PORT}/prompt",
                json={
                    "prompt": workflow,
                    "client_id": client_id,
                },
                timeout=30,
            )
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]
            print(f"[job:{job_id[:8]}] Prompt submitted: {prompt_id}")

            # 6. Wait for completion via WebSocket
            total_nodes = len(workflow)
            done_nodes = set()

            while True:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    # Poll history as fallback
                    try:
                        hr = req.get(
                            f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}",
                            timeout=5,
                        )
                        hist = hr.json().get(prompt_id, {})
                        status_obj = hist.get("status", {})
                        status_str = (
                            status_obj.get("status", "")
                            if isinstance(status_obj, dict)
                            else str(status_obj)
                        ).lower()
                        if status_str in ("success", "completed"):
                            break
                        if status_str in ("error", "failed"):
                            raise RuntimeError(f"ComfyUI execution failed: {status_obj}")
                    except RuntimeError:
                        raise
                    except Exception:
                        pass
                    continue

                if isinstance(raw, bytes):
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")
                data = msg.get("data", {})

                if data.get("prompt_id") != prompt_id:
                    continue

                if msg_type == "executing":
                    node = data.get("node")
                    if node is None:
                        print(f"[job:{job_id[:8]}] Execution complete")
                        break
                    done_nodes.add(str(node))

                elif msg_type == "execution_error":
                    raise RuntimeError(f"Execution error: {data}")

                elif msg_type in ("executed", "execution_cached"):
                    node = data.get("node")
                    if node:
                        done_nodes.add(str(node))

                # Update progress
                progress = int(len(done_nodes) / max(total_nodes, 1) * 100)
                if progress != job_data.get("progress", 0):
                    job_data["progress"] = min(progress, 99)
                    self._save_job(job_file, job_data, commit=False)

            ws.close()

            # 7. Collect outputs and upload to R2
            outputs = []
            hr = req.get(
                f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}",
                timeout=30,
            )
            history = hr.json().get(prompt_id, {})

            for node_id, node_out in history.get("outputs", {}).items():
                for media_type in ("images", "videos", "gifs", "audio"):
                    for file_info in node_out.get(media_type, []):
                        fr = req.get(
                            f"http://127.0.0.1:{COMFY_PORT}/view",
                            params={
                                "filename": file_info["filename"],
                                "subfolder": file_info.get("subfolder", ""),
                                "type": file_info.get("type", "output"),
                            },
                            timeout=120,
                            stream=True,
                        )
                        payload = b"".join(fr.iter_content(1 << 20))
                        fr.close()

                        # Upload to R2
                        ext = Path(file_info["filename"]).suffix.lower()
                        content_type = self._content_type(ext)
                        r2_key = f"outputs/{job_id}/{file_info['filename']}"

                        try:
                            upload_to_r2(payload, r2_key, content_type)
                            url = generate_r2_url(r2_key, expires_in=86400)
                            outputs.append({
                                "filename": file_info["filename"],
                                "type": media_type.rstrip("s"),
                                "size_bytes": len(payload),
                                "url": url,
                                "r2_key": r2_key,
                            })
                        except Exception as e:
                            print(f"[job:{job_id[:8]}] R2 upload failed, falling back to base64: {e}")
                            outputs.append({
                                "filename": file_info["filename"],
                                "type": media_type.rstrip("s"),
                                "size_bytes": len(payload),
                                "data": base64.b64encode(payload).decode(),
                            })

                # Text outputs
                for text_item in node_out.get("text", []):
                    outputs.append({
                        "node_id": node_id,
                        "type": "text",
                        "data": text_item,
                    })

            # 8. Mark completed
            job_data["status"] = "completed"
            job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            job_data["progress"] = 100
            job_data["outputs"] = outputs
            self._save_job(job_file, job_data)

            # 9. Send webhook if configured
            if job_data.get("webhook_url"):
                self._send_webhook(job_data["webhook_url"], job_id, "job.completed")

            print(f"[job:{job_id[:8]}] Completed with {len(outputs)} output(s)")

        except Exception as e:
            print(f"[job:{job_id[:8]}] Failed: {e}")
            job_data["status"] = "failed"
            job_data["error"] = str(e)
            job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            self._save_job(job_file, job_data)

            if job_data.get("webhook_url"):
                self._send_webhook(job_data["webhook_url"], job_id, "job.failed")
            raise

        finally:
            shutil.rmtree(input_dir, ignore_errors=True)

    # -- helpers --

    def _save_job(self, path: Path, data: dict, commit: bool = True):
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        if commit:
            jobs_vol.commit()

    def _remap(self, obj: Any, mapping: dict) -> Any:
        if isinstance(obj, dict):
            return {k: self._remap(v, mapping) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._remap(v, mapping) for v in obj]
        if isinstance(obj, str) and obj in mapping:
            return mapping[obj]
        return obj

    def _content_type(self, ext: str) -> str:
        types = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".gif": "image/gif", ".webp": "image/webp",
            ".mp4": "video/mp4", ".webm": "video/webm",
            ".wav": "audio/wav", ".mp3": "audio/mpeg",
        }
        return types.get(ext, "application/octet-stream")

    def _send_webhook(self, url: str, job_id: str, event: str):
        import httpx

        payload = {
            "event": event,
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        for attempt in range(3):
            try:
                with httpx.Client(timeout=10) as c:
                    c.post(url, json=payload).raise_for_status()
                return
            except Exception:
                if attempt == 2:
                    print(f"[webhook] Failed after 3 attempts for {job_id}")
                time.sleep(2 ** attempt)


# ---------------------------------------------------------------------------
# FastAPI application (runs on lightweight CPU containers)
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Request, Depends  # noqa: E402
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # noqa: E402

web_app = FastAPI(title="ComfyUI SaaS API", version="3.0.0")
security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Validate the Bearer token against API_KEY secret."""
    import os

    expected = os.environ.get("API_KEY", "")
    if not expected or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@web_app.post("/v1/jobs", response_model=JobCreateResponse)
async def create_job(body: JobCreate, _key: str = Depends(verify_api_key)):
    """Submit a ComfyUI workflow job."""
    if not body.workflow:
        raise HTTPException(400, "Workflow cannot be empty")
    if len(body.media) > 10:
        raise HTTPException(400, "Maximum 10 media files per job")
    for m in body.media:
        if m.data and len(m.data) * 3 / 4 > 50 * 1024 * 1024:
            raise HTTPException(400, f"Media '{m.name}' exceeds 50MB limit")

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    job_data = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "workflow": body.workflow,
        "inputs": [i.model_dump() for i in body.inputs],
        "media": [m.model_dump() for m in body.media],
        "webhook_url": body.webhook_url,
        "progress": 0,
        "outputs": [],
    }

    job_file = Path(JOBS_DIR) / f"{job_id}.json"
    job_file.parent.mkdir(parents=True, exist_ok=True)
    job_file.write_text(json.dumps(job_data))
    jobs_vol.commit()

    # Dispatch to GPU worker
    ComfyService().run_job.spawn(job_id)

    return JobCreateResponse(job_id=job_id, status=JobStatus.QUEUED, created_at=now)


@web_app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, _key: str = Depends(verify_api_key)):
    """Get job status and outputs."""
    job_file = Path(JOBS_DIR) / f"{job_id}.json"
    jobs_vol.reload()

    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    data = json.loads(job_file.read_text())
    return JobStatusResponse(
        job_id=data["job_id"],
        status=data["status"],
        created_at=data["created_at"],
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        progress=data.get("progress", 0),
        outputs=data.get("outputs", []),
        error=data.get("error"),
    )


@web_app.get("/v1/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    _key: str = Depends(verify_api_key),
):
    """List recent jobs."""
    limit = min(limit, 200)
    jobs_path = Path(JOBS_DIR)
    jobs_vol.reload()

    results = []
    if jobs_path.exists():
        files = sorted(jobs_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        for jf in files[:limit]:
            try:
                d = json.loads(jf.read_text())
                if status and d["status"] != status:
                    continue
                results.append({
                    "job_id": d["job_id"],
                    "status": d["status"],
                    "created_at": d["created_at"],
                    "progress": d.get("progress", 0),
                })
            except Exception:
                continue

    return {"jobs": results}


@web_app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str, _key: str = Depends(verify_api_key)):
    """Cancel a job (marks as failed, GPU may still finish current step)."""
    job_file = Path(JOBS_DIR) / f"{job_id}.json"
    jobs_vol.reload()

    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    data = json.loads(job_file.read_text())
    if data["status"] in ("completed", "failed"):
        return {"message": "Job already finished"}

    data["status"] = "failed"
    data["error"] = "Cancelled by user"
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    job_file.write_text(json.dumps(data))
    jobs_vol.commit()

    return {"message": "Job cancelled"}


@web_app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Deploy the API on lightweight containers
# ---------------------------------------------------------------------------


@app.function(
    image=api_image,
    volumes={JOBS_DIR: jobs_vol},
    secrets=[api_secret],
    max_containers=5,
    scaledown_window=60,
)
@modal.asgi_app()
def api():
    return web_app


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


@app.function(image=gpu_image, volumes={CACHE_DIR: cache_vol}, timeout=300)
def verify_setup():
    """Check models and custom nodes. Run: modal run comfyui_api.py::verify_setup"""
    models = list(Path(f"{CACHE_DIR}/models").rglob("*.safetensors"))
    total_gb = sum(m.stat().st_size for m in models) / (1024 ** 3)
    print(f"Models: {len(models)} files, {total_gb:.1f} GB total")

    nodes_dir = Path(f"{COMFY_DIR}/custom_nodes")
    if nodes_dir.exists():
        nodes = [d.name for d in nodes_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        print(f"Custom nodes: {len(nodes)}")
        for n in sorted(nodes):
            print(f"  - {n}")

    print("\nSetup OK!")


@app.function(
    schedule=modal.Cron("0 3 * * *"),
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={JOBS_DIR: jobs_vol},
)
def cleanup_old_jobs():
    """Delete job files older than 7 days."""
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    jobs_path = Path(JOBS_DIR)
    jobs_vol.reload()
    deleted = 0

    if jobs_path.exists():
        for jf in jobs_path.glob("*.json"):
            try:
                d = json.loads(jf.read_text())
                created = datetime.fromisoformat(d["created_at"])
                if created < cutoff:
                    jf.unlink()
                    deleted += 1
            except Exception:
                continue

    if deleted:
        jobs_vol.commit()
        print(f"Cleaned up {deleted} old jobs")
