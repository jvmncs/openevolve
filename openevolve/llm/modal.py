import io
import json
import os
import pathlib
import subprocess

import modal

from openevolve.utils.modal_utils import InferenceConfig

APP_NAME = "openevolve-llm"
app = modal.App(APP_NAME)
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    .run_commands(
        "pip install -U uv",
        "uv pip install --system huggingface_hub[hf_xet]",
        "uv pip install --system --torch-backend=cu128 vllm==0.9.1",
        "uv pip install --system https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl",
    )
    .env(
        {
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
            "VLLM_USE_V1_ENGINE": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "NCCL_CUMEM_ENABLE": "1",
        }
    )
)
bigmodel_cache = modal.Volume.from_name("big-model-hfcache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("openevolve-vllm-cache", create_if_missing=True)
flashinfer_cache = modal.Volume.from_name(
    "openevolve-flashinfer-cache", create_if_missing=True
)
config_vol = modal.Volume.from_name(
    "openevolve-vllm-config-cache", create_if_missing=True
)
inference_secret = modal.Secret.from_name("openevolve-vllm-secret")


class Inference:
    """Serve a model via vLLM with readiness check."""

    @modal.enter()
    def _startup(self):
        """Start the vLLM server and block until it is healthy."""
        import time
        import urllib.request

        api_key = os.environ.get("VLLM_API_KEY")

        with open("/root/config/vllm_config.json") as f:
            cfg = json.load(f)

        model_id = cfg.get("MODEL_ID")
        model_rev = cfg.get("MODEL_REV")
        vllm_cfg = cfg.get("VLLM_CFG")

        if model_id is None:
            raise ValueError(
                "MODEL_ID envvar is not set; can't start inference server without a model."
            )

        cmd = [
            "vllm",
            "serve",
            "--uvicorn-log-level=info",
            model_id,
            "--host",
            "0.0.0.0",
            "--port",
            "8081",
            "--api-key",
            api_key,
        ]
        if model_rev is not None:
            cmd.extend(["--revision", model_rev])
        if vllm_cfg is not None:
            cmd.extend(vllm_cfg.split(" "))

        subprocess.Popen(cmd)

        url: str = "http://127.0.0.1:8081/health"
        deadline: float = time.time() + 30 * 60
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url) as response:
                    if response.status == 200:
                        print("Server is healthy 🚀 –", url)
                        return
            except Exception:
                pass
            time.sleep(5)

        raise RuntimeError("Health-check failed – server did not respond in time")

    @modal.web_server(
        port=8081,
        startup_timeout=24 * 60 * 60,  # superceded by cls timeout setting
    )
    def serve(self):
        """Web server endpoint (actual server started in _startup)."""
        return

    @modal.method()
    def boot(self):
        # no-op for cold-booting
        pass


def build_inference_server(cfg: InferenceConfig):
    vllm_full_cfg = {
        "MODEL_ID": cfg.model_id,
        "MODEL_REV": cfg.model_rev,
    }
    if cfg.vllm_cfg is not None and len(cfg.vllm_cfg) > 0:
        vllm_full_cfg["VLLM_CFG"] = " ".join(cfg.vllm_cfg)

    config_bytes = io.BytesIO(json.dumps(vllm_full_cfg, indent=2).encode("utf-8"))
    with config_vol.batch_upload(force=True) as v:
        v.put_file(config_bytes, "vllm_config.json")

    cls_opts = {
        "image": inference_image,
        "gpu": cfg.deployment_config.gpu_str(),
        "volumes": {
            "/root/.cache/vllm": vllm_cache,
            "/root/.cache/flashinfer": flashinfer_cache,
            "/root/.cache/huggingface": bigmodel_cache,
            "/root/config": config_vol,
        },
        "secrets": [inference_secret],
    }

    if cfg.timeout is not None:
        cls_opts["timeout"] = cfg.timeout
    if cfg.scaledown_window is not None:
        cls_opts["scaledown_window"] = cfg.scaledown_window

    return app.cls(**cls_opts)(
        modal.concurrent(
            max_inputs=cfg.max_input_concurrency,
            target_inputs=cfg.target_input_concurrency,
        )(Inference)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy OpenEvolve LLM inference server"
    )
    parser.add_argument(
        "--config", required=True, help="Path to inference configuration YAML file"
    )

    args = parser.parse_args()
    config = args.config
    # Read config
    config_path = pathlib.Path(config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading configuration from {config_path}...")
    inference_config = InferenceConfig.from_yaml(config_path)
    name = inference_config.deployment_name
    if inference_config.deployment_tags is not None:
        tag = "-".join(inference_config.deployment_tags)
    else:
        tag = ""

    print(f"Building inference service for model {inference_config.model_id}...")
    inference_service = build_inference_server(inference_config)

    with modal.enable_output():
        print(f"Deploying app as '{name or APP_NAME}' with 'tag={tag or 'None'}'")
        app.deploy(name=name, tag=tag)

        # Boot the service
        print("Booting inference service...")
        inference_service().boot.spawn()
        print("Boot spawned successfully. Monitor in Modal dashboard.")
