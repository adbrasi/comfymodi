# build_sage_wheel.py
import modal
import os
from pathlib import Path


# Define an image that has all the build tools and the correct PyTorch version
builder_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.12")
    # Install g++, git, and the ninja build system
    .apt_install("git", "build-essential", "ninja-build")
    # Install PyTorch >= 2.3.0 for SageAttention compatibility. Let's use the latest for cu121.
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu128"
    )
    # Other build dependencies
    .pip_install("huggingface_hub", "setuptools", "wheel")
)

app = modal.App("sage-builder")

@app.function(
    image=builder_image,
    gpu="L40S", # Match the GPU of your main service
    secrets=[modal.Secret.from_name("hf-token")], # Needs a secret with your HF_TOKEN
    timeout=1800, # 30 minutes, compilation can be slow
)
def build_and_upload():
    """
    Clones SageAttention, builds the Python wheel using g++, and uploads it to Hugging Face Hub.
    """
    SAGE_REPO = "https://github.com/thu-ml/SageAttention.git"
    WORK_DIR = Path("/root/build")
    WORK_DIR.mkdir(exist_ok=True)

    print("Cloning SageAttention repository...")
    # Use --depth 1 for a faster clone
    os.system(f"git clone --depth 1 {SAGE_REPO} {WORK_DIR / 'SageAttention'}")

    build_dir = WORK_DIR / "SageAttention"
    os.chdir(build_dir)

    print("Building the wheel for SageAttention...")

    # --- KEY CHANGE HERE ---
    # Set environment variables to force the use of g++ for both C and C++ compilation/linking
    # This overrides the system's default which was trying to use clang++
    build_command = "CC=gcc CXX=g++ python setup.py bdist_wheel"
    print(f"Running build command: {build_command}")

    build_process = os.system(build_command)
    # -----------------------

    if build_process != 0:
        raise RuntimeError("Wheel building failed!")

    print("Wheel built successfully.")

    # Find the generated wheel file
    dist_dir = build_dir / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        raise RuntimeError("No wheel file found after build.")

    wheel_path = wheels[0]
    print(f"Found wheel: {wheel_path}")

    # Upload to Hugging Face Hub
    from huggingface_hub import HfApi
    api = HfApi()

    # REMEMBER TO CHANGE THIS
    hf_repo_id = "adbrasi/comfywheel"

    print(f"Uploading {wheel_path.name} to {hf_repo_id}...")

    api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(wheel_path),
        path_in_repo=wheel_path.name,
        repo_id=hf_repo_id,
    )

    print(f"✅ Successfully uploaded wheel to Hugging Face Hub!")
    print(f"You can now install it using: pip install https://huggingface.co/{hf_repo_id}/resolve/main/{wheel_path.name}")