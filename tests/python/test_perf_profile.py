import os
import subprocess
import sys

import pytest

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, "..", ".."))
python_root = os.path.join(root, "python")
examples_root = os.path.join(root, "examples")


def _run_perf_profile(device: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = python_root
    cmd = [sys.executable, os.path.join(examples_root, "perf_profile.py"), device, "8", "8", "8", "2"]
    subprocess.run(cmd, check=True, env=env, cwd=root, capture_output=True, text=True)


def test_perf_profile_cpu():
    _run_perf_profile("cpu")


def test_perf_profile_cuda_if_available():
    sys.path.insert(0, python_root)
    from tinyfin import backend_set
    if not backend_set("cuda"):
        pytest.skip("cuda backend not available")
    _run_perf_profile("cuda")
