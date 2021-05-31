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


def test_cast(backend):
    target = np.random.random(10)
    final = op.to_numpy(op.cast(target))
    np.testing.assert_allclose(final, target)
