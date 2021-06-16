import pytest
import numpy as np
from qibojit import custom_operators as op


def test_backend_setter(backend_name):
    original_backend = op.get_backend()
    op.set_backend(backend_name)
    assert op.backend.name == backend_name
    with pytest.raises(KeyError):
        op.set_backend("test")
    op.set_backend(original_backend)


@pytest.mark.parametrize("array_type", [None, "float32", "float64"])
def test_cast(backend, array_type):
    target = np.random.random(10)
    final = op.to_numpy(op.cast(target, dtype=array_type))
    np.testing.assert_allclose(final, target)


def test_to_numpy(backend):
    x = [0, 1, 2]
    target = op.to_numpy(op.cast(x))
    if op.get_backend() == "numba":
        final = op.to_numpy(x)
    else: # pragma: no cover
        final = op.to_numpy(np.array(x))
    np.testing.assert_allclose(final, target)
