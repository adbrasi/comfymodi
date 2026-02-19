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
    # Copy the memory_snapshot_helper custom node (patches ComfyUI for safe snapshotting)
    # Source: https://github.com/modal-labs/modal-examples/tree/main/06_gpu_and_ml/comfyui/memory_snapshot
    .add_local_dir(
        local_path=Path(__file__).parent / "memory_snapshot_helper",
        remote_path=f"{COMFY_DIR}/custom_nodes/memory_snapshot_helper",
        copy=True,
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
    CANCELLED = "cancelled"


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
    # Optional caller-supplied identifier for multi-tenant tracking.
    # Your SaaS backend should pass your internal user/org ID here.
    user_id: Optional[str] = None


class JobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    user_id: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    user_id: Optional[str] = None
    # Progress (0-100)
    progress: int = 0
    # Sampler step progress (e.g., diffusion step 15/30)
    current_step: int = 0
    total_steps: int = 0
    # Node-level progress
    current_node: Optional[str] = None
    nodes_done: int = 0
    nodes_total: int = 0
    # Results
    outputs: List[Dict] = Field(default_factory=list)
    error: Optional[str] = None
    # Execution log (timestamps + events for debugging)
    logs: List[str] = Field(default_factory=list)


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
    scaledown_window=300,  # keep warm 5 min after last job
    max_containers=10,
    timeout=600,
)
@modal.concurrent(max_inputs=1)
class ComfyService:
    """Manages a ComfyUI server and processes workflow jobs.

    Memory snapshot strategy (from official Modal ComfyUI example):
    - snap=True:  launch ComfyUI in background (CPU only, no CUDA) → snapshot saved
    - snap=False: restore CUDA device via /cuda/set_device → ready in seconds
    """

    @modal.enter(snap=True)
    def launch_for_snapshot(self):
        """Launch ComfyUI server during snapshot phase (no GPU).
        The memory_snapshot_helper custom node patches ComfyUI to avoid CUDA init here.
        """
        print("[snapshot] Launching ComfyUI for snapshotting ...")
        cmd = f"comfy launch --background -- --port {COMFY_PORT}"
        subprocess.run(cmd, shell=True, check=True)
        print("[snapshot] ComfyUI launched, snapshot will be captured")

    @modal.enter(snap=False)
    def restore_cuda(self):
        """After snapshot restore, re-enable the CUDA device for inference."""
        import requests as req

        print("[restore] Re-enabling CUDA device ...")
        try:
            r = req.post(f"http://127.0.0.1:{COMFY_PORT}/cuda/set_device", timeout=10)
            if r.status_code == 200:
                print("[restore] CUDA device ready!")
            else:
                print(f"[restore] Warning: /cuda/set_device returned {r.status_code}")
        except Exception as e:
            print(f"[restore] Warning: could not set CUDA device: {e}")

    def _check_health(self):
        """Check if ComfyUI is responsive; remove this container from pool if not."""
        import requests as req

        try:
            r = req.get(f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=5)
            r.raise_for_status()
        except Exception as e:
            print(f"[health] ComfyUI unhealthy: {e} — stopping container")
            modal.experimental.stop_fetching_inputs()
            raise RuntimeError("ComfyUI not healthy, container removed from pool")

    @modal.method()
    def run_job(self, job_id: str):
        """Execute a queued job end-to-end with detailed progress tracking."""
        import requests as req
        import websocket
        import shutil

        short_id = job_id[:8]

        def log(msg: str):
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            line = f"[{ts}] {msg}"
            print(f"[job:{short_id}] {line}")
            job_data.setdefault("logs", []).append(line)

        self._check_health()

        job_file = Path(JOBS_DIR) / f"{job_id}.json"
        input_dir = Path(f"{COMFY_DIR}/input") / job_id
        job_data: dict = {}

        try:
            # 1. Load job data
            jobs_vol.reload()
            if not job_file.exists():
                raise FileNotFoundError(f"Job {job_id} not found")
            job_data = json.loads(job_file.read_text())

            # Guard: check if job was cancelled/timed-out while GPU was spinning up
            if job_data.get("status") in ("cancelled", "failed"):
                print(f"[job:{short_id}] Job already {job_data['status']} before execution — skipping")
                return

            job_data["logs"] = []

            # 2. Mark as running
            job_data.update({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "progress": 0,
                "current_node": None,
                "current_step": 0,
                "total_steps": 0,
                "nodes_done": 0,
                "nodes_total": 0,
            })
            self._save_job(job_file, job_data)
            log("Job started")

            workflow = json.loads(json.dumps(job_data["workflow"]))

            # 3. Prepare media files
            input_dir.mkdir(parents=True, exist_ok=True)
            media_remap: dict = {}

            for item in job_data.get("media", []):
                name = item.get("name", f"media_{uuid.uuid4().hex[:8]}")
                safe_name = Path(name).name
                dest = input_dir / safe_name

                if item.get("data"):
                    dest.write_bytes(base64.b64decode(item["data"]))
                    log(f"Media loaded from base64: {safe_name}")
                elif item.get("url"):
                    r = req.get(item["url"], timeout=30)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                    log(f"Media downloaded: {safe_name} ({len(r.content)//1024}KB)")

                media_remap[safe_name] = str(Path(job_id) / safe_name)

            if media_remap:
                workflow = self._remap(workflow, media_remap)

            # 4. Apply input overrides
            for inp in job_data.get("inputs", []):
                node_id = str(inp["node"])
                if node_id not in workflow:
                    continue
                workflow.setdefault(node_id, {}).setdefault("inputs", {})
                workflow[node_id]["inputs"][inp["field"]] = inp["value"]
                log(f"Override node={node_id} field={inp['field']}")

            # 5. Submit to ComfyUI via WebSocket
            client_id = uuid.uuid4().hex
            ws = websocket.create_connection(
                f"ws://127.0.0.1:{COMFY_PORT}/ws?clientId={client_id}",
                timeout=15,
            )
            ws.settimeout(10)

            resp = req.post(
                f"http://127.0.0.1:{COMFY_PORT}/prompt",
                json={"prompt": workflow, "client_id": client_id},
                timeout=30,
            )
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]
            log(f"Prompt submitted: {prompt_id}")

            # 6. Track progress via WebSocket events
            #
            # Progress formula:
            #   80% = node execution (each completed node = 80/total%)
            #   +up to 80/total% = sampler steps within current node
            #   20% = output collection/upload phase
            #
            node_order: list = []          # execution order from ComfyUI
            done_nodes: set = set()        # completed node IDs
            cached_nodes: set = set()
            current_node_id: str = ""
            current_step: int = 0
            total_steps: int = 0
            last_commit = time.time()
            last_cancel_check = time.time()

            def calc_progress() -> int:
                n = max(len(node_order) or len(workflow), 1)
                node_pct = len(done_nodes) / n * 80.0
                step_pct = 0.0
                if total_steps > 0 and current_node_id and current_node_id not in done_nodes:
                    step_pct = (current_step / total_steps) * (1 / n) * 80.0
                return max(0, min(99, int(node_pct + step_pct)))

            def flush(force: bool = False):
                nonlocal last_commit
                job_data["progress"] = calc_progress()
                job_data["current_node"] = current_node_id or None
                job_data["current_step"] = current_step
                job_data["total_steps"] = total_steps
                job_data["nodes_done"] = len(done_nodes)
                job_data["nodes_total"] = len(node_order) or len(workflow)
                now = time.time()
                should_commit = force or (now - last_commit >= 3)
                self._save_job(job_file, job_data, commit=should_commit)
                if should_commit:
                    last_commit = now

            while True:
                # Periodic cancellation check (every ~15 s) — reads the job
                # file from the volume so the cancel_job endpoint can signal us.
                _t = time.time()
                if _t - last_cancel_check > 15:
                    last_cancel_check = _t
                    try:
                        jobs_vol.reload()
                        cur_status = json.loads(job_file.read_text()).get("status")
                        if cur_status in ("cancelled", "failed"):
                            log(f"Job {cur_status} mid-execution — stopping")
                            ws.close()
                            return
                    except Exception:
                        pass

                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    # Fallback: poll history to detect completion
                    try:
                        hr = req.get(
                            f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}",
                            timeout=5,
                        )
                        hist = hr.json().get(prompt_id, {})
                        s = hist.get("status", {})
                        s_str = (s.get("status", "") if isinstance(s, dict) else str(s)).lower()
                        if s_str in ("success", "completed"):
                            log("Detected completion via history poll")
                            break
                        if s_str in ("error", "failed"):
                            raise RuntimeError(f"ComfyUI reported failure: {s}")
                    except RuntimeError:
                        raise
                    except Exception:
                        pass
                    flush()
                    continue

                # Skip binary preview frames
                if isinstance(raw, bytes):
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")
                data = msg.get("data", {})

                # Only handle events for our prompt
                if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                    continue

                if msg_type == "execution_start":
                    # ComfyUI sends the planned execution order
                    nodes = data.get("nodes") or []
                    node_order = [str(n) for n in nodes if n is not None]
                    job_data["nodes_total"] = len(node_order)
                    log(f"Execution started — {len(node_order)} nodes planned")
                    flush(force=True)

                elif msg_type == "executing":
                    node = data.get("node")
                    if node is None:
                        log("All nodes executed — workflow complete")
                        break
                    current_node_id = str(node)
                    current_step = 0
                    total_steps = 0
                    node_type = workflow.get(current_node_id, {}).get("class_type", "?")
                    log(f"Executing node {current_node_id} ({node_type})")
                    flush(force=True)

                elif msg_type == "executed":
                    node = data.get("node")
                    if node:
                        done_nodes.add(str(node))
                        log(f"Node {node} completed ({len(done_nodes)}/{len(node_order) or len(workflow)})")
                    flush(force=True)

                elif msg_type == "execution_cached":
                    node = data.get("node")
                    if node:
                        cached_nodes.add(str(node))
                        done_nodes.add(str(node))
                        log(f"Node {node} cached (skipped)")
                    flush()

                elif msg_type == "progress":
                    # Sampler step progress (e.g., diffusion steps 1..30)
                    current_step = int(data.get("value", 0))
                    total_steps = int(data.get("max", 0))
                    flush()

                elif msg_type == "execution_error":
                    err = data.get("exception_message", str(data))
                    log(f"Execution error: {err}")
                    raise RuntimeError(f"ComfyUI execution error: {err}")

            ws.close()
            log("WebSocket closed")

            # 7. Collect outputs and upload to R2
            log("Collecting outputs ...")
            outputs = []
            hr = req.get(
                f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}",
                timeout=30,
            )
            history = hr.json().get(prompt_id, {})

            for node_id, node_out in history.get("outputs", {}).items():
                for media_type in ("images", "videos", "gifs", "audio"):
                    for file_info in node_out.get(media_type, []):
                        fname = file_info["filename"]
                        fr = req.get(
                            f"http://127.0.0.1:{COMFY_PORT}/view",
                            params={
                                "filename": fname,
                                "subfolder": file_info.get("subfolder", ""),
                                "type": file_info.get("type", "output"),
                            },
                            timeout=120,
                            stream=True,
                        )
                        payload = b"".join(fr.iter_content(1 << 20))
                        fr.close()

                        ext = Path(fname).suffix.lower()
                        r2_key = f"outputs/{job_id}/{fname}"

                        try:
                            upload_to_r2(payload, r2_key, self._content_type(ext))
                            url = generate_r2_url(r2_key, expires_in=86400)
                            outputs.append({
                                "filename": fname,
                                "type": media_type.rstrip("s"),
                                "size_bytes": len(payload),
                                "url": url,
                                "r2_key": r2_key,
                            })
                            log(f"Uploaded {fname} to R2 ({len(payload)//1024}KB)")
                        except Exception as e:
                            log(f"R2 upload failed for {fname}: {e} — using base64 fallback")
                            outputs.append({
                                "filename": fname,
                                "type": media_type.rstrip("s"),
                                "size_bytes": len(payload),
                                "data": base64.b64encode(payload).decode(),
                            })

                for text_item in node_out.get("text", []):
                    outputs.append({"node_id": node_id, "type": "text", "data": text_item})

            # 8. Mark completed
            elapsed = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(job_data["started_at"])
            ).total_seconds()
            log(f"Completed! {len(outputs)} output(s) | elapsed: {elapsed:.1f}s")

            job_data.update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "progress": 100,
                "current_node": None,
                "nodes_done": len(done_nodes),
                "outputs": outputs,
            })
            self._save_job(job_file, job_data)

            if job_data.get("webhook_url"):
                self._send_webhook(job_data["webhook_url"], job_id, "job.completed")

        except Exception as e:
            log(f"FAILED: {e}")
            job_data.update({
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
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
        if m.url:
            try:
                _validate_external_url(m.url, field=f"media '{m.name}' url")
            except ValueError as e:
                raise HTTPException(400, str(e))

    if body.webhook_url:
        try:
            _validate_external_url(body.webhook_url, field="webhook_url")
        except ValueError as e:
            raise HTTPException(400, str(e))

    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    job_data = {
        "job_id": job_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "user_id": body.user_id,
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

    return JobCreateResponse(job_id=job_id, status=JobStatus.QUEUED, created_at=now, user_id=body.user_id)


QUEUED_TIMEOUT_SECONDS = 120  # 2 min — if still queued, GPU unavailable → fail


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


def _validate_external_url(url: str, field: str = "URL") -> str:
    """Validate a user-supplied URL is safe to fetch (SSRF prevention).

    Blocks:
    - Non-http(s) schemes (file://, ftp://, etc.)
    - Private RFC-1918 ranges (10.x, 172.16-31.x, 192.168.x)
    - Loopback (127.x, ::1)
    - Link-local (169.254.x, fe80::)
    - Reserved / unspecified addresses
    """
    import ipaddress
    import urllib.parse

    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        raise ValueError(f"Malformed {field}: {url!r}")

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"{field} must use http or https (got {parsed.scheme!r})")

    host = parsed.hostname
    if not host:
        raise ValueError(f"{field} is missing a hostname")

    # Reject bare IP addresses in private/loopback ranges
    try:
        ip = ipaddress.ip_address(host)
        if any([
            ip.is_private, ip.is_loopback, ip.is_link_local,
            ip.is_reserved, ip.is_unspecified, ip.is_multicast,
        ]):
            raise ValueError(f"{field} resolves to a restricted IP range — not allowed")
    except ValueError as exc:
        if "restricted" in str(exc):
            raise
        # host is a domain name, not a raw IP — that's fine

    return url


def _check_queued_timeout(data: dict) -> dict:
    """If a job has been queued too long (GPU unavailable), auto-fail it."""
    if data.get("status") != "queued":
        return data
    created = datetime.fromisoformat(data["created_at"])
    elapsed = (datetime.now(timezone.utc) - created).total_seconds()
    if elapsed > QUEUED_TIMEOUT_SECONDS:
        data["status"] = "failed"
        data["error"] = f"Job timed out after {elapsed:.0f}s in queue (GPU unavailable)"
        data["completed_at"] = datetime.now(timezone.utc).isoformat()
        job_file = Path(JOBS_DIR) / f"{data['job_id']}.json"
        job_file.write_text(json.dumps(data))
        jobs_vol.commit()
    return data


@web_app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str, _key: str = Depends(verify_api_key)):
    """Get job status and outputs."""
    job_file = Path(JOBS_DIR) / f"{job_id}.json"
    jobs_vol.reload()

    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    data = _check_queued_timeout(json.loads(job_file.read_text()))
    return JobStatusResponse(
        job_id=data["job_id"],
        status=data["status"],
        created_at=data["created_at"],
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        user_id=data.get("user_id"),
        progress=data.get("progress", 0),
        current_step=data.get("current_step", 0),
        total_steps=data.get("total_steps", 0),
        current_node=data.get("current_node"),
        nodes_done=data.get("nodes_done", 0),
        nodes_total=data.get("nodes_total", 0),
        outputs=data.get("outputs", []),
        error=data.get("error"),
        logs=data.get("logs", []),
    )


@web_app.get("/v1/jobs")
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    user_id: Optional[str] = None,
    _key: str = Depends(verify_api_key),
):
    """List recent jobs. Filter by status and/or user_id for multi-tenant isolation."""
    limit = min(limit, 200)
    jobs_path = Path(JOBS_DIR)
    jobs_vol.reload()

    results = []
    if jobs_path.exists():
        files = sorted(jobs_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        for jf in files[:limit]:
            try:
                d = _check_queued_timeout(json.loads(jf.read_text()))
                if status and d["status"] != status:
                    continue
                if user_id and d.get("user_id") != user_id:
                    continue
                results.append({
                    "job_id": d["job_id"],
                    "status": d["status"],
                    "created_at": d["created_at"],
                    "progress": d.get("progress", 0),
                    "user_id": d.get("user_id"),
                })
            except Exception:
                continue

    return {"jobs": results}


@web_app.delete("/v1/jobs/{job_id}")
async def cancel_job(job_id: str, _key: str = Depends(verify_api_key)):
    """Cancel a queued or running job.

    Sets status to 'cancelled' in the volume. The GPU worker polls this flag
    every ~15 seconds and will stop cleanly on the next cycle.
    """
    job_file = Path(JOBS_DIR) / f"{job_id}.json"
    jobs_vol.reload()

    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    data = json.loads(job_file.read_text())
    if data["status"] in ("completed", "failed", "cancelled"):
        return {"message": f"Job already {data['status']}"}

    data["status"] = "cancelled"
    data["error"] = "Cancelled by user"
    data["completed_at"] = datetime.now(timezone.utc).isoformat()
    job_file.write_text(json.dumps(data))
    jobs_vol.commit()

    return {"message": "Job cancelled — worker will stop within ~15 seconds"}


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
