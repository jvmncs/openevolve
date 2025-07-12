import dataclasses
import enum
import json
import pathlib
import secrets
import subprocess
from typing import Union, Optional

import yaml


class ModalGPU(enum.StrEnum):
    H100 = "H100"
    H200 = "H200"
    A100_40GB = "A100-40GB"
    A100_80GB = "A100-80GB"
    B200 = "B200"
    L40S = "L40S"


@dataclasses.dataclass
class ModalClusterConfig:
    num_nodes: int
    gpus_per_node: int
    gpu_type: Union[str, ModalGPU] = ModalGPU.H100

    def __post_init__(self):
        if isinstance(self.gpu_type, str):
            try:
                self.gpu_type = ModalGPU(self.gpu_type)
            except ValueError:
                valid_gpu_types = ", ".join([f"'{g.value}'" for g in ModalGPU])
                raise ValueError(
                    f"Invalid GPU type '{self.gpu_type}'. Must be one of: {valid_gpu_types}"
                )

        # TODO remove when @modal.experimental.clustered supports more GPU types
        if self.gpu_type != ModalGPU.H100 and self.num_nodes != 1:
            raise ValueError(
                f"num_nodes must be 1 when using gpu_type {self.gpu_type}. "
                f"At time of writing, only {ModalGPU.H100} supports multiple nodes."
            )

    def gpu_str(self):
        return f"{self.gpu_type}:{self.gpus_per_node}"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModalClusterConfig":
        """Create configuration from a dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary"""
        return {
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "gpu_type": (
                self.gpu_type.value
                if isinstance(self.gpu_type, ModalGPU)
                else self.gpu_type
            ),
        }


@dataclasses.dataclass
class InferenceConfig:
    model_id: str
    model_rev: str
    target_input_concurrency: int
    max_input_concurrency: int
    vllm_cfg: list[str] = dataclasses.field(default_factory=lambda: list())
    deployment_config: ModalClusterConfig = dataclasses.field(
        default_factory=lambda: ModalClusterConfig(
            num_nodes=1, gpus_per_node=1, gpu_type=ModalGPU.H100
        )
    )
    timeout: Optional[int] = None
    scaledown_window: Optional[int] = None
    deployment_name: Optional[str] = None
    deployment_tags: Optional[list[str]] = None

    @classmethod
    def from_yaml(cls, path: Union[str, pathlib.Path]) -> "InferenceConfig":
        """Load configuration from a YAML file"""

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict) -> "InferenceConfig":
        """Create configuration from a dictionary"""
        # Handle nested deployment_config
        if "deployment_config" in config_dict:
            config_dict["deployment_config"] = ModalClusterConfig.from_dict(
                config_dict["deployment_config"]
            )
        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to a dictionary"""
        return {
            "model_id": self.model_id,
            "model_rev": self.model_rev,
            "target_input_concurrency": self.target_input_concurrency,
            "max_input_concurrency": self.max_input_concurrency,
            "vllm_cfg": self.vllm_cfg,
            "deployment_config": self.deployment_config.to_dict(),
            "timeout": self.timeout,
            "scaledown_window": self.scaledown_window,
            "deployment_name": self.deployment_name,
            "deployment_tags": self.deployment_tags,
        }

    def to_yaml(self, path: Union[str, pathlib.Path]) -> None:
        """Save configuration to a YAML file"""

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def ensure_vllm_secret(secret_name: str, force: bool = False):
    # bit of a hack to get around the Python client being incomplete
    try:
        result = subprocess.run(
            ["modal", "secret", "list", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        secrets_list = json.loads(result.stdout)
        secret_exists = any(s["Name"] == secret_name for s in secrets_list)

        if not secret_exists or force:
            vllm_api_key = secrets.token_urlsafe(32)
            print(f"Creating secret '{secret_name}' with VLLM_API_KEY...")
            create_cmd = [
                "modal",
                "secret",
                "create",
                secret_name,
                f"VLLM_API_KEY={vllm_api_key}",
            ]
            if force:
                create_cmd.append("--force")
            subprocess.run(create_cmd, check=True)
            print(f"Secret '{secret_name}' created successfully.")
            cwd = pathlib.Path.cwd()
            local_secret_path = cwd / f"{secret_name}.secret"
            print(f"Saving secret locally to {local_secret_path}...")
            local_secret_path.write_text(vllm_api_key)
            print("Secret saved.")
            return vllm_api_key
        else:
            print(f"Secret '{secret_name}' already exists. Skipping creation.")

    except FileNotFoundError:
        print(
            "Error: 'modal' command not found. Please ensure Modal CLI is installed and in your PATH."
        )
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error running modal command: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output from modal command: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create or check VLLM secret")
    parser.add_argument(
        "secret_name",
        nargs="?",
        default="openevolve-vllm-secret",
        help="Name of the secret to create or check (default: openevolve-vllm-secret)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force creation of the secret even if it already exists",
    )

    args = parser.parse_args()
    ensure_vllm_secret(args.secret_name, args.force)
