import json
import os
import subprocess
import sys


def _run_backend_env(name: str) -> dict:
    here = os.path.dirname(__file__)
    root = os.path.normpath(os.path.join(here, "..", "..", "python"))
    env = os.environ.copy()
    env["PYTHONPATH"] = root
    env["TINYFIN_BACKEND"] = name
    code = (
        "import json\n"
        "from tinyfin import backend_name, backend_set\n"
        "name = backend_name()\n"
        "cuda_ok = backend_set('cuda')\n"
        "probe_ok = backend_set(%r)\n"
        "print(json.dumps({'name': name, 'cuda_ok': bool(cuda_ok), 'probe_ok': bool(probe_ok)}))\n"
    ) % name
    res = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(res.stdout.strip())


def test_backend_env_cpu():
    info = _run_backend_env("cpu")
    assert info["name"] == "cpu"


def test_backend_env_unknown_falls_back_to_cpu():
    info = _run_backend_env("nonexistent_backend")
    assert info["name"] == "cpu"


def test_backend_env_cuda_if_available():
    info = _run_backend_env("cuda")
    if info["cuda_ok"]:
        assert info["name"] == "cuda"
    else:
        assert info["name"] == "cpu"


def test_backend_env_blas_if_available():
    info = _run_backend_env("blas")
    if info["probe_ok"]:
        assert info["name"] == "blas"
    else:
        assert info["name"] == "cpu"


def test_backend_env_opengl_if_available():
    info = _run_backend_env("opengl")
    if info["probe_ok"]:
        assert info["name"] == "opengl"
    else:
        assert info["name"] == "cpu"


def test_backend_env_vulkan_if_available():
    info = _run_backend_env("vulkan")
    if info["probe_ok"]:
        assert info["name"] == "vulkan"
    else:
        assert info["name"] == "cpu"
