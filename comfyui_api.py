"""
ComfyUI SaaS API on Modal.com

REST API that executes any ComfyUI workflow on GPU.
Outputs uploaded to Cloudflare R2, returned as signed URLs.

Deploy:  modal deploy comfyui_api.py
Test:    python test_run.py
"""

import json
import uuid
import base64
import time
import subprocess
import os
import hmac
import ipaddress
import urllib.parse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

import modal
import modal.experimental

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "comfyui-saas"
COMFY_PORT = 8188

CACHE_VOL_NAME = "comfyui-models-cache"
JOBS_VOL_NAME = "comfyui-jobs"

CACHE_DIR = "/cache"
JOBS_DIR = "/jobs"
COMFY_DIR = "/root/comfy/ComfyUI"

GPU_MAX_CONTAINERS = int(os.environ.get("GPU_MAX_CONTAINERS", "2"))
GPU_MIN_CONTAINERS = int(os.environ.get("GPU_MIN_CONTAINERS", "0"))
GPU_BUFFER_CONTAINERS = int(os.environ.get("GPU_BUFFER_CONTAINERS", "0"))
GPU_SCALEDOWN_WINDOW_SECONDS = int(os.environ.get("GPU_SCALEDOWN_WINDOW_SECONDS", "60"))
API_MAX_CONTAINERS = int(os.environ.get("API_MAX_CONTAINERS", "1"))
MAX_ACTIVE_JOBS_PER_USER = int(os.environ.get("MAX_ACTIVE_JOBS_PER_USER", "5"))
QUEUED_TIMEOUT_SECONDS = int(os.environ.get("QUEUED_TIMEOUT_SECONDS", "360"))
R2_URL_TTL_SECONDS = int(os.environ.get("R2_URL_TTL_SECONDS", "86400"))


def _parse_gpu_config() -> str | list[str]:
    raw = os.environ.get("GPU_CONFIG", "l40s,a100,a10g")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) <= 1:
        return parts[0] if parts else "l40s"
    return parts


GPU_CONFIG = _parse_gpu_config()

# Models: (hf_repo_id, hf_filename, target_subdir_under_models/)
MODELS = [
    ("OnomaAIResearch/Illustrious-XL-v1.0", "Illustrious-XL-v1.0.safetensors", "checkpoints"),
]

# Custom nodes: list of git URLs (add commit hash as tuple for pinning)
CUSTOM_NODES: list = []

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)

cache_vol = modal.Volume.from_name(CACHE_VOL_NAME, create_if_missing=True)
jobs_vol = modal.Volume.from_name(JOBS_VOL_NAME, create_if_missing=True)

api_secret = modal.Secret.from_name("comfyui-api-secret", required_keys=["API_KEY"])
r2_secret = modal.Secret.from_name(
    "comfyui-r2",
    required_keys=["R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL", "R2_BUCKET_NAME"],
)

# ---------------------------------------------------------------------------
# Image build helpers
# ---------------------------------------------------------------------------


def download_models():
    from huggingface_hub import hf_hub_download

    for repo_id, filename, subdir in MODELS:
        target_dir = Path(f"{CACHE_DIR}/models/{subdir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / Path(filename).name

        if target_path.exists() and target_path.stat().st_size > 1_000_000:
            print(f"[models] {filename} already cached, skipping")
            continue

        print(f"[models] Downloading {repo_id}/{filename} ...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(target_dir), local_dir_use_symlinks=False)
        print(f"[models] OK {filename}")

    for d in ["outputs", "temp"]:
        Path(f"{CACHE_DIR}/{d}").mkdir(parents=True, exist_ok=True)
    cache_vol.commit()


def install_custom_nodes():
    if not CUSTOM_NODES:
        return
    nodes_dir = Path(f"{COMFY_DIR}/custom_nodes")
    for entry in CUSTOM_NODES:
        if isinstance(entry, tuple):
            url, commit = entry
        else:
            url, commit = entry, None
        name = url.rstrip("/").split("/")[-1]
        dest = nodes_dir / name
        if dest.exists():
            continue
        print(f"[nodes] Installing {name} ...")
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True, timeout=120)
        if commit:
            subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit], check=True, timeout=60)
            subprocess.run(["git", "-C", str(dest), "checkout", commit], check=True, timeout=30)
        req = dest / "requirements.txt"
        if req.exists():
            subprocess.run(["pip", "install", "-q", "-r", str(req)], check=False, timeout=120)


# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "curl", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
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
    .run_commands("comfy --skip-prompt install --skip-manager --fast-deps --nvidia")
    .add_local_dir(
        local_path=Path(__file__).parent / "memory_snapshot_helper",
        remote_path=f"{COMFY_DIR}/custom_nodes/memory_snapshot_helper",
        copy=True,
    )
    .run_function(install_custom_nodes)
    .run_commands(
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
    .pip_install("fastapi[standard]", "pydantic", "boto3")
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
    data: Optional[str] = None
    url: Optional[str] = None


class WorkflowOverride(BaseModel):
    node: str
    field: str
    value: Any
    type: str = "raw"


class JobCreate(BaseModel):
    workflow: Dict
    inputs: List[WorkflowOverride] = Field(default_factory=list)
    media: List[MediaInput] = Field(default_factory=list)
    webhook_url: Optional[str] = None
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
    progress: int = 0
    current_step: int = 0
    total_steps: int = 0
    current_node: Optional[str] = None
    nodes_done: int = 0
    nodes_total: int = 0
    outputs: List[Dict] = Field(default_factory=list)
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# R2 helpers
# ---------------------------------------------------------------------------


def _r2_client():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_to_r2(file_bytes: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    _r2_client().put_object(Bucket=os.environ["R2_BUCKET_NAME"], Key=key, Body=file_bytes, ContentType=content_type)
    return key


def generate_r2_url(key: str, expires_in: int = 3600) -> str:
    return _r2_client().generate_presigned_url(
        "get_object", Params={"Bucket": os.environ["R2_BUCKET_NAME"], "Key": key}, ExpiresIn=expires_in
    )


# ---------------------------------------------------------------------------
# URL validation (basic SSRF prevention)
# ---------------------------------------------------------------------------


def _validate_url(url: str):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use http or https")
    host = (parsed.hostname or "").lower()
    if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or host.endswith(".local"):
        raise ValueError("URL points to a restricted host")
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError("URL points to a private IP")
    except ValueError as e:
        if "private" in str(e) or "restricted" in str(e) or "loopback" in str(e):
            raise


# ---------------------------------------------------------------------------
# GPU Worker: ComfyService
# ---------------------------------------------------------------------------


@app.cls(
    gpu=GPU_CONFIG,
    image=gpu_image,
    volumes={CACHE_DIR: cache_vol, JOBS_DIR: jobs_vol},
    secrets=[r2_secret],
    enable_memory_snapshot=True,
    scaledown_window=GPU_SCALEDOWN_WINDOW_SECONDS,
    max_containers=GPU_MAX_CONTAINERS,
    min_containers=GPU_MIN_CONTAINERS,
    buffer_containers=GPU_BUFFER_CONTAINERS,
    timeout=600,
)
@modal.concurrent(max_inputs=1)
class ComfyService:

    @modal.enter(snap=True)
    def launch_for_snapshot(self):
        print("[snapshot] Launching ComfyUI for snapshotting ...")
        subprocess.run(f"comfy launch --background -- --port {COMFY_PORT}", shell=True, check=True)
        print("[snapshot] ComfyUI launched, snapshot will be captured")

    @modal.enter(snap=False)
    def restore_cuda(self):
        import requests as req

        print("[restore] Re-enabling CUDA device ...")
        try:
            r = req.post(f"http://127.0.0.1:{COMFY_PORT}/cuda/set_device", timeout=10)
            print(f"[restore] CUDA device ready! (status={r.status_code})")
        except Exception as e:
            print(f"[restore] Warning: could not set CUDA device: {e}")

    def _check_health(self):
        import requests as req

        try:
            req.get(f"http://127.0.0.1:{COMFY_PORT}/system_stats", timeout=5).raise_for_status()
        except Exception as e:
            print(f"[health] ComfyUI unhealthy: {e} — stopping container")
            modal.experimental.stop_fetching_inputs()
            raise RuntimeError("ComfyUI not healthy")

    @modal.method()
    def run_job(self, job_id: str):
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
            # Load job
            jobs_vol.reload()
            if not job_file.exists():
                raise FileNotFoundError(f"Job {job_id} not found")
            job_data = json.loads(job_file.read_text())

            # Check if cancelled while GPU was spinning up
            if job_data.get("status") in ("cancelled", "failed"):
                print(f"[job:{short_id}] Job already {job_data['status']} — skipping")
                return

            # Mark as running
            job_data.update({
                "status": "running",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "progress": 0,
                "logs": [],
            })
            self._save_job(job_file, job_data)
            log("Job started")

            workflow = json.loads(json.dumps(job_data["workflow"]))

            # Prepare media files
            input_dir.mkdir(parents=True, exist_ok=True)
            media_remap: dict = {}
            for item in job_data.get("media", []):
                name = item.get("name", f"media_{uuid.uuid4().hex[:8]}")
                safe_name = Path(name).name
                dest = input_dir / safe_name
                if item.get("data"):
                    dest.write_bytes(base64.b64decode(item["data"]))
                    log(f"Media loaded: {safe_name}")
                elif item.get("url"):
                    import requests as dl_req

                    r = dl_req.get(item["url"], timeout=30)
                    r.raise_for_status()
                    dest.write_bytes(r.content)
                    log(f"Media downloaded: {safe_name} ({len(r.content) // 1024}KB)")
                media_remap[safe_name] = str(Path(job_id) / safe_name)

            if media_remap:
                workflow = self._remap(workflow, media_remap)

            # Apply input overrides
            for inp in job_data.get("inputs", []):
                node_id = str(inp["node"])
                if node_id in workflow:
                    workflow.setdefault(node_id, {}).setdefault("inputs", {})
                    workflow[node_id]["inputs"][inp["field"]] = inp["value"]
                    log(f"Override node={node_id} field={inp['field']}")

            # Submit to ComfyUI
            client_id = uuid.uuid4().hex
            ws = websocket.create_connection(
                f"ws://127.0.0.1:{COMFY_PORT}/ws?clientId={client_id}", timeout=15
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

            # Track progress via WebSocket
            node_order: list = []
            done_nodes: set = set()
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
                # Periodic cancellation check (every 30s)
                _t = time.time()
                if _t - last_cancel_check > 30:
                    last_cancel_check = _t
                    try:
                        jobs_vol.reload()
                        cur = json.loads(job_file.read_text())
                        if cur.get("status") in ("cancelled", "failed"):
                            log("Job cancelled mid-execution — stopping")
                            ws.close()
                            return
                    except Exception:
                        pass

                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    try:
                        hr = req.get(f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}", timeout=5)
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

                if isinstance(raw, bytes):
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")
                data = msg.get("data", {})

                if data.get("prompt_id") and data["prompt_id"] != prompt_id:
                    continue

                if msg_type == "execution_start":
                    nodes = data.get("nodes") or []
                    node_order = [str(n) for n in nodes if n is not None]
                    log(f"Execution started — {len(node_order)} nodes")
                    flush(force=True)

                elif msg_type == "executing":
                    node = data.get("node")
                    if node is None:
                        log("All nodes executed")
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
                        log(f"Node {node} done ({len(done_nodes)}/{len(node_order) or len(workflow)})")
                    flush(force=True)

                elif msg_type == "execution_cached":
                    node = data.get("node")
                    if node:
                        done_nodes.add(str(node))
                    flush()

                elif msg_type == "progress":
                    current_step = int(data.get("value", 0))
                    total_steps = int(data.get("max", 0))
                    flush()

                elif msg_type == "execution_error":
                    err = data.get("exception_message", str(data))
                    log(f"Execution error: {err}")
                    raise RuntimeError(f"ComfyUI execution error: {err}")

            ws.close()

            # Collect outputs and upload to R2
            log("Collecting outputs ...")
            outputs = []
            hr = req.get(f"http://127.0.0.1:{COMFY_PORT}/history/{prompt_id}", timeout=30)
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
                            url = generate_r2_url(r2_key, expires_in=R2_URL_TTL_SECONDS)
                            outputs.append({
                                "filename": fname,
                                "type": media_type.rstrip("s"),
                                "size_bytes": len(payload),
                                "url": url,
                                "r2_key": r2_key,
                            })
                            log(f"Uploaded {fname} ({len(payload) // 1024}KB)")
                        except Exception as e:
                            log(f"R2 upload failed for {fname}: {e}")
                            outputs.append({
                                "filename": fname,
                                "type": media_type.rstrip("s"),
                                "size_bytes": len(payload),
                                "error": f"Upload failed: {e}",
                            })

                for text_item in node_out.get("text", []):
                    outputs.append({"node_id": node_id, "type": "text", "data": text_item})

            # Mark completed
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(job_data["started_at"])).total_seconds()
            log(f"Completed! {len(outputs)} output(s) | {elapsed:.1f}s")

            job_data.update({
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "progress": 100,
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

        payload = {"event": event, "job_id": job_id, "timestamp": datetime.now(timezone.utc).isoformat()}
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
# FastAPI (lightweight CPU containers)
# ---------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Depends  # noqa: E402
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # noqa: E402

web_app = FastAPI(title="ComfyUI SaaS API", version="4.0.0")
security = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    expected = os.environ.get("API_KEY", "")
    if not expected or not hmac.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@web_app.post("/v1/jobs", response_model=JobCreateResponse)
def create_job(body: JobCreate, _key: str = Depends(verify_api_key)):
    if not body.workflow:
        raise HTTPException(400, "Workflow cannot be empty")
    if len(body.media) > 10:
        raise HTTPException(400, "Maximum 10 media files per job")

    for m in body.media:
        if m.data and len(m.data) * 3 / 4 > 50 * 1024 * 1024:
            raise HTTPException(400, f"Media '{m.name}' exceeds 50MB limit")
        if m.url:
            try:
                _validate_url(m.url)
            except ValueError as e:
                raise HTTPException(400, str(e))

    if body.webhook_url:
        try:
            _validate_url(body.webhook_url)
        except ValueError as e:
            raise HTTPException(400, str(e))

    # Simple per-user job count (no locks needed)
    if body.user_id:
        jobs_vol.reload()
        jobs_path = Path(JOBS_DIR)
        active_count = 0
        if jobs_path.exists():
            for jf in jobs_path.glob("*.json"):
                try:
                    d = json.loads(jf.read_text())
                    if d.get("user_id") == body.user_id and d.get("status") in ("queued", "running"):
                        active_count += 1
                except Exception:
                    continue
        if active_count >= MAX_ACTIVE_JOBS_PER_USER:
            raise HTTPException(429, "Too many active jobs. Wait for current jobs to finish.")

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
        "function_call_id": None,
    }

    try:
        job_file = Path(JOBS_DIR) / f"{job_id}.json"
        job_file.parent.mkdir(parents=True, exist_ok=True)
        job_file.write_text(json.dumps(job_data))
        jobs_vol.commit()

        fc = ComfyService().run_job.spawn(job_id)

        jobs_vol.reload()
        latest = json.loads(job_file.read_text())
        latest["function_call_id"] = fc.object_id
        job_file.write_text(json.dumps(latest))
        jobs_vol.commit()
    except Exception:
        try:
            job_file = Path(JOBS_DIR) / f"{job_id}.json"
            job_data["status"] = "failed"
            job_data["error"] = "Failed to enqueue GPU job"
            job_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            job_file.write_text(json.dumps(job_data))
            jobs_vol.commit()
        except Exception:
            pass
        raise HTTPException(503, "Could not dispatch job to worker")

    return JobCreateResponse(job_id=job_id, status=JobStatus.QUEUED, created_at=now, user_id=body.user_id)


@web_app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, _key: str = Depends(verify_api_key)):
    job_file = Path(JOBS_DIR) / f"{job_id}.json"
    jobs_vol.reload()
    if not job_file.exists():
        raise HTTPException(404, "Job not found")

    data = json.loads(job_file.read_text())

    # Refresh R2 URLs if outputs exist
    for out in data.get("outputs", []):
        if out.get("r2_key"):
            try:
                out["url"] = generate_r2_url(out["r2_key"], expires_in=R2_URL_TTL_SECONDS)
            except Exception:
                pass

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
def list_jobs(
    status: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 50,
    _key: str = Depends(verify_api_key),
):
    limit = min(limit, 200)
    jobs_path = Path(JOBS_DIR)
    jobs_vol.reload()

    results = []
    if jobs_path.exists():
        files = sorted(jobs_path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        for jf in files:
            try:
                d = json.loads(jf.read_text())
                if user_id and d.get("user_id") != user_id:
                    continue
                if status and d["status"] != status:
                    continue
                results.append({
                    "job_id": d["job_id"],
                    "status": d["status"],
                    "created_at": d["created_at"],
                    "progress": d.get("progress", 0),
                    "user_id": d.get("user_id"),
                })
                if len(results) >= limit:
                    break
            except Exception:
                continue

    return {"jobs": results}


@web_app.delete("/v1/jobs/{job_id}")
def cancel_job(job_id: str, _key: str = Depends(verify_api_key)):
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

    call_id = data.get("function_call_id")
    if call_id:
        try:
            modal.FunctionCall.from_id(call_id).cancel()
        except Exception:
            pass

    return {"message": "Job cancellation requested"}


@web_app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------


@app.function(
    image=api_image,
    volumes={JOBS_DIR: jobs_vol},
    secrets=[api_secret, r2_secret],
    max_containers=API_MAX_CONTAINERS,
    scaledown_window=60,
)
@modal.asgi_app()
def api():
    return web_app


# ---------------------------------------------------------------------------
# Utilities & scheduled tasks
# ---------------------------------------------------------------------------


@app.function(image=gpu_image, volumes={CACHE_DIR: cache_vol}, timeout=300)
def verify_setup():
    """Run: modal run comfyui_api.py::verify_setup"""
    models = list(Path(f"{CACHE_DIR}/models").rglob("*.safetensors"))
    total_gb = sum(m.stat().st_size for m in models) / (1024 ** 3)
    print(f"Models: {len(models)} files, {total_gb:.1f} GB")

    nodes_dir = Path(f"{COMFY_DIR}/custom_nodes")
    if nodes_dir.exists():
        nodes = [d.name for d in nodes_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        print(f"Custom nodes: {len(nodes)}")
        for n in sorted(nodes):
            print(f"  - {n}")
    print("\nSetup OK!")


@app.function(schedule=modal.Cron("*/1 * * * *"), image=api_image, volumes={JOBS_DIR: jobs_vol})
def fail_stale_queued_jobs():
    now = datetime.now(timezone.utc)
    jobs_path = Path(JOBS_DIR)
    jobs_vol.reload()
    changed = 0

    if jobs_path.exists():
        for jf in jobs_path.glob("*.json"):
            try:
                d = json.loads(jf.read_text())
            except Exception:
                continue
            if d.get("status") != "queued":
                continue
            elapsed = (now - datetime.fromisoformat(d["created_at"])).total_seconds()
            if elapsed <= QUEUED_TIMEOUT_SECONDS:
                continue
            d["status"] = "failed"
            d["error"] = f"Job timed out after {elapsed:.0f}s in queue"
            d["completed_at"] = now.isoformat()
            d["updated_at"] = now.isoformat()
            jf.write_text(json.dumps(d))
            changed += 1

    if changed:
        jobs_vol.commit()
        print(f"Marked {changed} queued jobs as failed")


@app.function(schedule=modal.Cron("0 3 * * *"), image=api_image, volumes={JOBS_DIR: jobs_vol})
def cleanup_old_jobs():
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    jobs_path = Path(JOBS_DIR)
    jobs_vol.reload()
    deleted = 0

    if jobs_path.exists():
        for jf in jobs_path.glob("*.json"):
            try:
                d = json.loads(jf.read_text())
                if datetime.fromisoformat(d["created_at"]) < cutoff:
                    jf.unlink()
                    deleted += 1
            except Exception:
                continue

    if deleted:
        jobs_vol.commit()
        print(f"Cleaned up {deleted} old jobs")
