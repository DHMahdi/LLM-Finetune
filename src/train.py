# train.py
import os
from datetime import datetime
import secrets as secrets_lib
from pathlib import Path

from .train_setup import (
    app,
    training_image,
    volume_manager,
    HOURS,
    MINUTES,
    GPU_CONFIG,
    SINGLE_GPU_CONFIG,
    run_cmd,
)

VOLUME_CONFIG = volume_manager.get_volume_config()

@app.function(
    image=training_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
)
def train(run_folder: str, output_dir: str):
    import torch
    print(f"Starting training run in {run_folder}.")
    print(f"Using {torch.cuda.device_count()} {torch.cuda.get_device_name()} GPU(s).")

    ALLOW_WANDB = os.environ.get("ALLOW_WANDB", "false").lower() == "true"
    cmd = f"accelerate launch -m axolotl.cli.train ./config.yml {'--wandb_mode disabled' if not ALLOW_WANDB else ''}"
    run_cmd(cmd, run_folder, VOLUME_CONFIG)

    # Kick off CPU job to merge the LoRA weights into base model
    merge_handle = merge.spawn(run_folder, output_dir)
    with open(f"{run_folder}/logs.txt", "a") as f:
        f.write(f"<br>merge: https://modal.com/logs/call/{merge_handle.object_id}\n")
        print(f"Beginning merge {merge_handle.object_id}.")
    return merge_handle

@app.function(
    image=training_image,
    gpu=SINGLE_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
)
def preproc_data(run_folder: str):
    print("Preprocessing data.")
    run_cmd(
        "python -W ignore:::torch.nn.modules.module -m axolotl.cli.preprocess ./config.yml",
        run_folder,
        VOLUME_CONFIG
    )

@app.function(
    image=training_image,
    gpu=SINGLE_GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=24 * HOURS,
)
def merge(run_folder: str, output_dir: str):
    import shutil

    output_path = Path(run_folder) / output_dir
    shutil.rmtree(output_path / "merged", ignore_errors=True)

    with open(f"{run_folder}/config.yml"):
        print(f"Merge from {output_path}")

    MERGE_CMD = f"accelerate launch -m axolotl.cli.merge_lora ./config.yml --lora_model_dir='{output_dir}'"
    run_cmd(MERGE_CMD, run_folder, VOLUME_CONFIG)

    VOLUME_CONFIG["/runs"].commit()

@app.function(
    image=training_image,
    timeout=30 * MINUTES,
    volumes=VOLUME_CONFIG
)
def launch(config_raw: dict, data_raw: str, run_to_resume: str, preproc_only: bool):
    import yaml
    from huggingface_hub import snapshot_download

    # Ensure the base model is downloaded
    config = yaml.safe_load(config_raw)
    model_name = config["base_model"]

    try:
        snapshot_download(model_name, local_files_only=True)
        print(f"Volume contains {model_name}.")
    except FileNotFoundError:
        print(f"Downloading {model_name} ...")
        snapshot_download(model_name)

        print("Committing /pretrained directory...")
        VOLUME_CONFIG["/pretrained"].commit()

    # Write config and data into a training subfolder
    time_string = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_name = (
        f"axo-{time_string}-{secrets_lib.token_hex(2)}"
        if not run_to_resume
        else run_to_resume
    )
    run_folder = f"/runs/{run_name}"
    os.makedirs(run_folder, exist_ok=True)

    print(f"Preparing training run in {run_folder}.")
    with (
        open(f"{run_folder}/config.yml", "w") as config_file,
        open(f"{run_folder}/{config['datasets'][0]['path']}", "w") as data_file,
    ):
        config_file.write(config_raw)
        data_file.write(data_raw)
    VOLUME_CONFIG["/runs"].commit()

    if preproc_only:
        print("Spawning container for data preprocessing.")
        launch_handle = preproc_data.spawn(run_folder)
    else:
        print("Spawning container for data preprocessing.")
        preproc_handle = preproc_data.spawn(run_folder)
        with open(f"{run_folder}/logs.txt", "w") as f:
            lbl = "preproc"
            f.write(f"{lbl}: https://modal.com/logs/call/{preproc_handle.object_id}")
        # wait for preprocessing to finish
        preproc_handle.get()

        # Start training run
        print("Spawning container for training.")
        launch_handle = train.spawn(run_folder, config["output_dir"])

    with open(f"{run_folder}/logs.txt", "w") as f:
        lbl = "train" if not preproc_only else "preproc"
        f.write(f"{lbl}: https://modal.com/logs/call/{launch_handle.object_id}")
    VOLUME_CONFIG["/runs"].commit()

    return run_name, launch_handle

@app.local_entrypoint()
def main(
    config: str,
    data: str,
    merge_lora: bool = True,
    preproc_only: bool = False,
    run_to_resume: str = None,
):
    # Read config and data source files with UTF-8 encoding
    with (
        open(config, "r", encoding="utf-8") as cfg,
        open(data, "r", encoding="utf-8") as dat
    ):
        run_name, launch_handle = launch.remote(
            cfg.read(), dat.read(), run_to_resume, preproc_only
        )

    # Write a local reference to the run location
    with open(".last_run_name", "w", encoding="utf-8") as f:
        f.write(run_name)

    # Wait for the training run to finish
    merge_handle = launch_handle.get()
    if merge_lora and not preproc_only:
        merge_handle.get()

    print(f"Run complete. Tag: {run_name}")
    print(f"To inspect outputs, run `modal volume ls training-runs-vol {run_name}`")
    if not preproc_only:
        print(
            f"To run sample inference, run `modal run -q src.inference --run-name {run_name}`"
        )