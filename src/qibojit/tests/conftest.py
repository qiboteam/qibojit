import pytest
import cupy as cp
from qibojit import custom_operators as op


BACKENDS = ["numba"]
try:
    if cp.cuda.runtime.getDeviceCount(): # test cupy backend in GPU is available
        BACKENDS.append("cupy")
except cp.cuda.runtime.CUDARuntimeError:
    pass


@pytest.fixture
def backend(backend_name):
    original_backend = op.get_backend()
    op.set_backend(backend_name)
    yield
    op.set_backend(original_backend)


def pytest_generate_tests(metafunc):
    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", BACKENDS)
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", ["complex128", "complex64"])
