# setup.py
import os
from pathlib import PurePosixPath
from typing import Union, Dict

import modal
from modal import Image, Secret, Volume

MINUTES = 60
HOURS = 60 * MINUTES

GPU_CONFIG = os.environ.get("GPU_CONFIG", "a100:2")
SINGLE_GPU_CONFIG = os.environ.get("GPU_CONFIG", "a10g:1")

class VolumeManager:
    def __init__(self):
        self.pretrained_volume = Volume.from_name(
            "pretrained-models-vol", 
            create_if_missing=True
        )
        self.runs_volume = Volume.from_name(
            "training-runs-vol", 
            create_if_missing=True
        )
        
    def get_volume_config(self) -> Dict[Union[str, PurePosixPath], Volume]:
        return {
            "/pretrained": self.pretrained_volume,
            "/runs": self.runs_volume
        }

def create_training_image() -> Image:
    """Create and return the Axolotl training image with LoRA support."""
    return (
        Image.from_registry(
            "winglian/axolotl@sha256:9578c47333bdcc9ad7318e54506b9adaf283161092ae780353d506f7a656590a"
        )
        .pip_install(
            "huggingface_hub==0.23.2",
            "hf-transfer==0.1.5",
            "wandb==0.16.3",
            "bitsandbytes>=0.41.1",
            "scipy",
            "fastapi==0.110.0",
            "pydantic==2.6.3",
        )
        .env({
            "HUGGINGFACE_HUB_CACHE": "/pretrained",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TQDM_DISABLE": "true",
            "AXOLOTL_NCCL_TIMEOUT": "60",
        })
        .entrypoint([])
    )

def get_secrets():
    """Set up and return required secrets."""
    return [
        Secret.from_name("my-huggingface-secret"),
        Secret.from_dict({
            "ALLOW_WANDB": os.environ.get("ALLOW_WANDB", "false")
        }),
    ]

# Initialize app with all configurations
volume_manager = VolumeManager()
training_image = create_training_image()
secrets = get_secrets()

app = modal.App(
    "sentiment-fine-tuning",
    secrets=secrets,
)

def run_cmd(cmd: str, run_folder: str, volume_config: dict):
    """Run a command inside a folder, with Modal Volume reloading before and commit on success."""
    import subprocess

    # Ensure volumes contain latest files
    volume_config["/pretrained"].reload()
    volume_config["/runs"].reload()

    # Propagate errors from subprocess
    if exit_code := subprocess.call(cmd.split(), cwd=run_folder):
        exit(exit_code)

    # Commit writes to volume
    volume_config["/runs"].commit()