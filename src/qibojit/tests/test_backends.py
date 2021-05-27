import pytest
from qibojit import custom_operators as op


def test_backend_setter(backend_name):
    original_backend = op.get_backend()
    op.set_backend(backend_name)
    assert op.backend.name == backend_name
    with pytest.raises(KeyError):
        op.set_backend("test")
    op.set_backend(original_backend)
