import pytest
import qibo
from qibo import K
qibo.set_backend("qibojit")


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
def platform(platform_name):
    original_platform = K.platform.name
    K.set_platform(platform_name)
    yield
    K.set_platform(original_platform)


def pytest_generate_tests(metafunc):
    if "platform_name" in metafunc.fixturenames:
        metafunc.parametrize("platform_name", K.available_platforms)
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("precision", ["double", "single"])
