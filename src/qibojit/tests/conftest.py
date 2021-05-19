import pytest
from qibojit import custom_operators as op


@pytest.fixture
def backend(backend_name):
    original_backend = op.get_backend()
    op.set_backend(backend_name)
    yield
    op.set_backend(original_backend)


def pytest_generate_tests(metafunc):
    # TODO: Fix `complex64` for cupy
    if "backend_name" in metafunc.fixturenames:
        if "dtype" in metafunc.fixturenames:
            metafunc.parametrize("backend_name,dtype",
                                 [("numba", "complex128"),
                                  ("numba", "complex64"),
                                  ("cupy", "complex128"),
                                  ("cupy", "complex64")])
        else:
            metafunc.parametrize("backend_name", ["numba", "cupy"])
    elif "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", ["complex128", "complex64"])
