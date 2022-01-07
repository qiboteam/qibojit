import pytest
import qibo
from qibo import K
qibo.set_backend("qibojit")

_BACKENDS = ["numba"]
if K.gpu_devices:  # pragma: no cover
    # CI does not test for GPU
    _BACKENDS.extend(("cupy", "cuquantum"))

@pytest.fixture
def dtype(precision):
    original_precision = qibo.get_precision()
    qibo.set_precision(precision)
    if precision == "double":
        yield "complex128"
    else:
        yield "complex64"
    qibo.set_precision(original_precision)

@pytest.fixture
def backend(backend_name):
    original_backend = K.engine.name
    K.set_engine(backend_name)
    yield
    K.set_engine(original_backend)


def pytest_generate_tests(metafunc):
    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", _BACKENDS)
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("precision", ["double", "single"])
