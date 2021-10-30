import pytest
import numpy as np
from qibo import K


def test_engine_setter(backend_name):
    original_backend = K.engine.name
    K.set_engine(backend_name)
    assert K.engine.name == backend_name
    with pytest.raises(ValueError):
        K.set_engine("test")
    K.set_engine(original_backend)


@pytest.mark.parametrize("array_type", [None, "float32", "float64"])
def test_cast(backend, array_type):
    target = np.random.random(10)
    final = K.to_numpy(K.cast(target, dtype=array_type))
    np.testing.assert_allclose(final, target)


def test_to_numpy(backend):
    x = [0, 1, 2]
    target = K.to_numpy(K.cast(x))
    if K.engine.name == "numba":
        final = K.to_numpy(x)
    else: # pragma: no cover
        final = K.to_numpy(np.array(x))
    np.testing.assert_allclose(final, target)
