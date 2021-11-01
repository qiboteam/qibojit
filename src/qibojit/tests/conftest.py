import pytest
import qibo
from qibo import K
qibo.set_backend("qibojit")

_BACKENDS = ["numba"]
if K._cupy_engine is not None:  # pragma: no cover
    # CI does not test for GPU
    _BACKENDS.append("cupy")


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
        metafunc.parametrize("dtype", ["complex128", "complex64"])
