# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI SaaS API deployment on Modal.com that provides a scalable, serverless REST API for ComfyUI workflows. The system transforms ComfyUI workflows into production-ready endpoints with GPU acceleration, job queuing, and webhook support.

## Common Development Commands

### Modal Deployment
```bash
# Deploy the main API service
python -m modal deploy modal_comfyui_with_sageattention.py

# Build and upload SageAttention wheel to Hugging Face
modal run build_whell.py

# Verify setup (check models and custom nodes)
modal run modal_comfyui_with_sageattention.py::verify_setup

# Monitor deployments
modal app list
```

### Testing the API
```bash
# Submit a job
curl -X POST https://<your-username>--comfyui-saas-apiLASTDANCE2-api.modal.run/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{"workflow": {...}}'

# Check job status
curl https://<your-username>--comfyui-saas-apiLASTDANCE2-api.modal.run/v1/jobs/{job_id}
```

## Architecture and Key Components

### Core Service Architecture
- **ComfyService Class** (`modal_comfyui_with_sageattention.py:233-646`): GPU-enabled container that manages ComfyUI execution with memory snapshots for fast cold starts (~8-10 seconds)
- **Two-phase initialization**: Snapshot creation without GPU (lines 235-265), then ComfyUI startup when GPU available (lines 270-332)
- **WebSocket-based progress tracking**: Real-time updates during workflow execution with proper binary frame handling

### Volume Management
- **comfyui-cache**: Persistent storage for models (~20GB), mounted at `/cache`
- **job-storage**: Job data persistence across containers at `/jobs`
- Models are symlinked from cache to ComfyUI directories for efficient access

### API Design
- FastAPI-based REST endpoints with async job processing
- Job queue with priority support and concurrent execution (max 5 jobs per container)
- Webhook notifications for job completion
- Base64-encoded media input/output for images and videos

### Performance Optimizations
- **Memory snapshots**: Pre-loaded Python libraries and environment setup
- **SageAttention integration**: Custom-built wheel for optimized attention mechanisms
- **Batch commits**: Efficient volume updates during job processing (10-second intervals)
- **Connection retry logic**: Robust ComfyUI server startup with 3 retry attempts

## Important Implementation Details

- The system uses L40S GPUs by default (configurable in line 221)
- Custom nodes are installed during image build for consistency
- Binary WebSocket frames from ComfyUI previews are properly handled (lines 426-434)
- Job files use JSON format for simplicity and debugging
- Automatic cleanup of old jobs runs daily via scheduled function