from tinyfin import backend_name, backend_set


def test_backend_set_known_names_preserve_state():
    for name in ("cpu", "cuda", "blas", "opengl", "vulkan"):
        before = backend_name()
        ok = backend_set(name)
        if ok:
            assert backend_name() == name
        else:
            assert backend_name() == before


def test_backend_set_unknown_keeps_backend():
    before = backend_name()
    assert backend_set("nonexistent_backend") is False
    assert backend_name() == before
