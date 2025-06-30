"""
Modal configuration for the function minimization example
"""

import pathlib
import modal

local_req_path = pathlib.Path(__file__).parent / "requirements.txt"
remote_req_path = pathlib.Path("/") / "root" / "requirements.txt"

# Define the image for this example
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(local_req_path, remote_req_path.as_posix(), copy=True)
    .run_commands(
        "uv pip install --system -U pip",
        f"uv pip install --system -r {remote_req_path}",
    )
)


# Export
sandbox_image = image
