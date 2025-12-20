"""Diagnose PennyLane GPU backend availability.

Run on the V100 machine (inside your Apptainer/uv env):

  uv run python -m src.check_pennylane_gpu

This script prints:
- torch / cuda info
- PennyLane + Lightning package versions
- available PennyLane devices
- whether `lightning.gpu` can be instantiated

If `lightning.gpu` fails, the error message here is the key to fixing performance.
"""

from __future__ import annotations

import os
import sys
import traceback


def _try_pkg_version(dist_name: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(dist_name)
    except Exception:
        return None


def main() -> None:
    print("=== Python ===")
    print(sys.version.replace("\n", " "))
    print("executable:", sys.executable)

    print("\n=== Environment (selected) ===")
    for k in [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        "LD_LIBRARY_PATH",
        "PATH",
    ]:
        v = os.environ.get(k)
        if v is not None:
            print(f"{k}={v}")

    print("\n=== PyTorch ===")
    try:
        import torch

        print("torch:", torch.__version__)
        print("cuda_available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0))
            print("torch.version.cuda:", torch.version.cuda)
            print("cudnn:", torch.backends.cudnn.version())
    except Exception:
        print("ERROR importing torch")
        traceback.print_exc()

    print("\n=== PennyLane / Lightning versions ===")
    for dist in [
        "pennylane",
        "pennylane-lightning",
        "pennylane-lightning-gpu",
        "pennylane_lightning",
        "pennylane_lightning_gpu",
    ]:
        ver = _try_pkg_version(dist)
        if ver:
            print(f"{dist}: {ver}")

    print("\n=== PennyLane devices ===")
    try:
        import pennylane as qml

        print("pennylane:", qml.__version__)
        print("qml.devices():", qml.devices())

        print("\nTry: qml.device('lightning.gpu', wires=2)")
        try:
            dev = qml.device("lightning.gpu", wires=2)
            print("SUCCESS, device:", dev)
        except Exception as e:
            print("FAILED:", type(e).__name__, e)

        print("\nTry: qml.device('lightning.qubit', wires=2)")
        try:
            dev = qml.device("lightning.qubit", wires=2)
            print("SUCCESS, device:", dev)
        except Exception as e:
            print("FAILED:", type(e).__name__, e)

    except Exception:
        print("ERROR importing pennylane")
        traceback.print_exc()


if __name__ == "__main__":
    main()
