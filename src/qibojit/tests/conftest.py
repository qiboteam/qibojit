import pytest
from qibojit import custom_operators as op


@pytest.fixture
def backend(backend_name):
    original_backend = op.get_backend()
    op.set_backend(backend_name)
    yield
    op.set_backend(original_backend)


def pytest_generate_tests(metafunc):
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", ["complex128", "complex64"])
    if "backend_name" in metafunc.fixturenames:
        metafunc.parametrize("backend_name", ["numba", "cupy"])
