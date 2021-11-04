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


def test_device_setter():
    original_device = K.default_device
    K.set_device("/CPU:0")
    assert K.default_device == "/CPU:0"
    K.set_device(original_device)


def test_thread_setter():
    import numba
    original_threads = numba.get_num_threads()
    K.set_threads(1)
    assert numba.get_num_threads() == 1
    K.set_threads(original_threads)


@pytest.mark.parametrize("array_type", [None, "float32", "float64"])
def test_cast(backend, array_type):
    target = np.random.random(10)
    final = K.to_numpy(K.cast(target, dtype=array_type))
    K.assert_allclose(final, target)


def test_to_numpy(backend):
    x = [0, 1, 2]
    target = K.to_numpy(K.cast(x))
    if K.engine.name == "numba":
        final = K.to_numpy(x)
    else: # pragma: no cover
        final = K.to_numpy(np.array(x))
    K.assert_allclose(final, target)


def test_basic_matrices(backend):
    K.assert_allclose(K.eye(4), np.eye(4))
    K.assert_allclose(K.zeros((3, 3)), np.zeros((3, 3)))
    K.assert_allclose(K.ones((5, 5)), np.ones((5, 5)))
    from scipy.linalg import expm
    m = np.random.random((4, 4))
    K.assert_allclose(K.expm(m), expm(m))


def test_backend_eigh(backend):
    m = np.random.random((16, 16))
    eigvals2, eigvecs2 = np.linalg.eigh(m)
    eigvals1, eigvecs1 = K.eigh(K.cast(m))
    K.assert_allclose(eigvals1, eigvals2)
    K.assert_allclose(K.abs(eigvecs1), np.abs(eigvecs2))


def test_backend_eigvalsh(backend):
    m = np.random.random((16, 16))
    target = np.linalg.eigvalsh(m)
    result = K.eigvalsh(K.cast(m))
    K.assert_allclose(target, result)


def test_unique_and_gather(backend):
    samples = np.random.randint(0, 2, size=(100,))
    K.assert_allclose(K.unique(samples), np.unique(samples))
    indices = [0, 2, 6, 40]
    final = K.gather(samples, indices)
    K.assert_allclose(final, samples[indices])


def test_with_device(backend):
    with K.device("/CPU:0"):
        pass
    with K.device("/GPU:0"):
        pass
    with K.on_cpu():
        pass
    target = np.random.random(5)
    final = K.cpu_cast(target)
    K.assert_allclose(final, target)


def test_cpu_ops(backend):
    with K.on_cpu():
        pass
