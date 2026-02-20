# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI SaaS API on Modal.com — a serverless REST API for executing ComfyUI workflows on GPU. Single file deployment (`comfyui_api.py`) with R2 storage, memory snapshots, and webhook support.

## Common Development Commands

```bash
# Deploy
modal deploy comfyui_api.py

# Verify models and custom nodes
modal run comfyui_api.py::verify_setup

# Test
export COMFYUI_API_KEY=your-key
export COMFYUI_API_URL=https://workspace--comfyui-saas-api.modal.run
python test_run.py

# Monitor
modal app list
modal app logs comfyui-saas
modal container list
```

## Architecture

- **`comfyui_api.py`**: Single file with all logic (~960 lines)
  - Config + constants (top of file)
  - Image build helpers (`download_models`, `install_custom_nodes`)
  - Two container images: `gpu_image` (PyTorch + ComfyUI) and `api_image` (lightweight FastAPI)
  - Pydantic models for API request/response
  - R2 helpers (upload, presigned URLs)
  - `ComfyService` class: GPU worker with memory snapshots, WebSocket progress tracking
  - FastAPI endpoints: CRUD for jobs + health check
  - Scheduled tasks: stale job timeout (1min cron), cleanup (daily 3am)

- **`memory_snapshot_helper/`**: Custom node that patches ComfyUI for safe snapshotting (no CUDA init during snapshot phase)

## Key Patterns

- **Memory snapshots**: `@modal.enter(snap=True)` launches ComfyUI CPU-only, `snap=False` restores CUDA via `/cuda/set_device`
- **Health check**: `modal.experimental.stop_fetching_inputs()` removes unhealthy containers
- **Job lifecycle**: queued → running → completed/failed/cancelled (JSON files on Modal Volume)
- **Cancellation**: `FunctionCall.cancel()` + cooperative check every 30s in WS loop
- **GPU fallback**: `GPU_CONFIG` env var supports comma-separated list (e.g., `l40s,a100,a10g`)
- **PyTorch pinned**: `torch==2.6.0`, `torchvision==0.21.0`, `torchaudio==2.6.0` (cu128)

## Important Notes

- Binary WebSocket frames from ComfyUI previews are skipped (not JSON)
- Job files are plain JSON on Modal Volume (`/jobs/{job_id}.json`)
- R2 URLs are refreshed on each `GET /v1/jobs/{id}` call
- `user_id` is optional in job creation (for tracking/filtering)
- Auth is Bearer token only (no X-User-ID header required)
