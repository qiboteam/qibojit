import pytest
import numpy as np
from qibojit.tests import backend as K


@pytest.mark.parametrize("array_type", [None, "float32", "float64"])
def test_cast(array_type):
    target = np.random.random(10)
    final = K.to_numpy(K.cast(target, dtype=array_type))
    np.testing.assert_allclose(final, target)


def test_to_numpy():
    x = [0, 1, 2]
    target = K.to_numpy(K.cast(x))
    if K.name == "numba":
        final = K.to_numpy(x)
    else: # pragma: no cover
        final = K.to_numpy(np.array(x))
    np.testing.assert_allclose(final, target)
