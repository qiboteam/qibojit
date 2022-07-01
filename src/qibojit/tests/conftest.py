import pytest
from qibojit.backends import NumbaBackend, CupyBackend

AVAILABLE_BACKENDS = ["numba", "cupy"]


@pytest.fixture
def backend(backend_name):
    if backend_name == "numba":
        yield NumbaBackend()
    elif backend_name == "cupy":
        yield CupyBackend()


def pytest_generate_tests(metafunc):
    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", AVAILABLE_BACKENDS)
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", ["complex128", "complex64"])
