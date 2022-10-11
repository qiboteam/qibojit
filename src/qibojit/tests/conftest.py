import pytest

from qibojit.backends import CupyBackend, CuQuantumBackend, NumbaBackend

BACKENDS = {"numba": NumbaBackend, "cupy": CupyBackend, "cuquantum": CuQuantumBackend}

# ignore backends that are not available in the current testing environment
AVAILABLE_BACKENDS = []
for backend_name in BACKENDS.keys():
    try:
        BACKENDS.get(backend_name)()
        AVAILABLE_BACKENDS.append(backend_name)
    except (ModuleNotFoundError, ImportError):
        pass


@pytest.fixture
def backend(backend_name):
    yield BACKENDS.get(backend_name)()


def pytest_generate_tests(metafunc):
    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", AVAILABLE_BACKENDS)
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", ["complex128", "complex64"])
