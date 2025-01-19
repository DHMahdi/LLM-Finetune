# ---
# deploy: true
# cmd: ["modal", "serve", "10_integrations/streamlit/serve_streamlit.py"]
# ---

# # Run and share Streamlit apps

# This example shows you how to run a Streamlit app with `modal serve`, and then deploy it as a serverless web app.

# ![example streamlit app](./streamlit.png)

# This example is structured as two files:

# 1. This module, which defines the Modal objects (name the script `serve_streamlit.py` locally).

# 2. `app.py`, which is any Streamlit script to be mounted into the Modal
# function ([download script](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/streamlit/app.py)).

import shlex
import subprocess
from pathlib import Path
import os
import modal
from modal import Secret, Volume

# Define volume configuration
runs_volume = Volume.from_name("training-runs-vol", create_if_missing=True)
pretrained_volume = Volume.from_name("pretrained-models-vol", create_if_missing=True)

streamlit_script_local_path = Path(__file__).parent / "app.py"
streamlit_script_remote_path = "/root/app.py"

image = (
    modal.Image.debian_slim(python_version="3.12.6")
    .run_commands("python -m pip install numpy pandas peft streamlit torch 'transformers>=4.45.1' vllm")
    .add_local_file(streamlit_script_local_path, streamlit_script_remote_path, copy=True)
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

secrets = get_secrets()
app = modal.App(name="sentiment-fine-tuning", image=image, secrets=secrets)

if not streamlit_script_local_path.exists():
    raise RuntimeError(
        "app.py not found! Place the script with your streamlit app in the same directory."
    )

@app.function(
    keep_warm=3,
    container_idle_timeout=600,
    allow_concurrent_inputs=100,
    volumes={
        "/runs": runs_volume,
        "/pretrained": pretrained_volume
    }
)
@modal.web_server(8000)
def run():
    target = shlex.quote(streamlit_script_remote_path)
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"
    subprocess.Popen(cmd, shell=True)


# ## Iterate and Deploy

# While you're iterating on your screamlit app, you can run it "ephemerally" with `modal serve`. This will
# run a local process that watches your files and updates the app if anything changes.

# ```shell
# modal serve serve_streamlit.py
# ```

# Once you're happy with your changes, you can deploy your application with

# ```shell
# modal deploy serve_streamlit.py
# ```

# If successful, this will print a URL for your app that you can navigate to from
# your browser 🎉 .