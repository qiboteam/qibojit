import numpy as np
import pytest
from qibo import set_device
from qibo.hamiltonians import TFIM
from scipy import sparse
from scipy.linalg import expm

from qibojit.backends import MetaBackend

from .conftest import AVAILABLE_BACKENDS, BACKENDS


def test_device_setter(backend):
    if backend.platform == "numba":
        device = "/CPU:0"

        with pytest.raises(ValueError):
            set_device("/CPU:1")

    else:
        device = "/GPU:0"
    backend.set_device(device)
    assert backend.device == device


def test_thread_setter(backend):
    import numba

    original_threads = numba.get_num_threads()
    backend.set_threads(1)
    assert numba.get_num_threads() == 1
    backend.set_threads(original_threads)


@pytest.mark.parametrize("array_type", [None, "float32", "float64"])
def test_cast(backend, array_type):
    target = np.random.random(10)
    final = backend.to_numpy(backend.cast(target, dtype=array_type))
    backend.assert_allclose(final, target)


@pytest.mark.parametrize("array_type", [None, "float32", "float64"])
@pytest.mark.parametrize("format", ["coo", "csr", "csc", "dia"])
def test_sparse_cast(backend, array_type, format):
    sptarget = sparse.rand(512, 512, dtype=array_type, format=format)
    assert backend.is_sparse(sptarget)
    final = backend.to_numpy(backend.cast(sptarget))
    target = sptarget.toarray()
    backend.assert_allclose(final, target)
    if backend.platform != "numba":
        sptarget = getattr(backend.sparse, sptarget.__class__.__name__)(sptarget)
        assert backend.is_sparse(sptarget)
        final = backend.to_numpy(backend.cast(sptarget))
        backend.assert_allclose(final, target)


def test_to_numpy(backend):
    x = [0, 1, 2]
    target = backend.to_numpy(backend.cast(x))
    if backend.platform == "numba":
        final = backend.to_numpy(x)
    else:
        final = backend.to_numpy(np.array(x))
    backend.assert_allclose(final, target)


@pytest.mark.parametrize("sparse_type", ["coo", "csr", "csc", "dia"])
def test_backend_expm_sparse(backend, sparse_type):
    if backend.platform == "numba" and sparse_type == "dia":
        pytest.skip("Problems with numba and dia.")

    m = sparse.rand(16, 16, format=sparse_type)
    target = expm(m.toarray())
    result = backend.to_numpy(backend.matrix_exp(backend.cast(m, dtype=m.dtype)))
    backend.assert_allclose(target, result, atol=1e-10)


@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_backend_eigh(backend, sparse_type):
    if sparse_type is None:
        m = np.random.random((16, 16))
        eigvals1, eigvecs1 = backend.eigenvectors(backend.cast(m, dtype=m.dtype))
        eigvals2, eigvecs2 = np.linalg.eigh(m)
    else:
        m = sparse.rand(16, 16, format=sparse_type)
        m = m + m.T
        eigvals1, eigvecs1 = backend.eigenvectors(backend.cast(m), k=16)
        eigvals2, eigvecs2 = backend.eigenvectors(backend.cast(m.toarray()))
    backend.assert_allclose(eigvals1, eigvals2, atol=1e-10)
    eigvecs1 = backend.to_numpy(eigvecs1)
    eigvecs2 = backend.to_numpy(eigvecs2)
    backend.assert_allclose(np.abs(eigvecs1), np.abs(eigvecs2), atol=1e-10)


@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_backend_eigvalsh(backend, sparse_type):
    if sparse_type is None:
        m = np.random.random((16, 16))
        target = np.linalg.eigvalsh(m)
        result = backend.eigenvalues(backend.cast(m))
    else:
        m = sparse.rand(16, 16, format=sparse_type)
        m = m + m.T
        result = backend.eigenvalues(backend.cast(m), k=16)
        target, _ = backend.eigenvectors(backend.cast(m.toarray()))
    backend.assert_allclose(target, result, atol=1e-10)


@pytest.mark.parametrize("sparse_type", ["coo", "csr", "csc", "dia"])
@pytest.mark.parametrize("k", [6, 8])
def test_backend_eigh_sparse(backend, sparse_type, k):
    ham = TFIM(6, h=1.0, backend=backend)
    m = getattr(sparse, f"{sparse_type}_matrix")(backend.to_numpy(ham.matrix))
    eigvals1, _ = backend.eigenvectors(backend.cast(m), k)
    eigvals2, _ = sparse.linalg.eigsh(m, k, which="SA")
    eigvals1 = backend.to_numpy(eigvals1)
    eigvals2 = backend.to_numpy(eigvals2)
    backend.assert_allclose(sorted(eigvals1), sorted(eigvals2))


def test_metabackend_list_available():
    available_backends = {
        backend: backend in AVAILABLE_BACKENDS for backend in BACKENDS
    }
    assert MetaBackend().list_available() == available_backends
