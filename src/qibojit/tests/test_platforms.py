import pytest
import numpy as np
from qibo import K


def test_platform_setter_and_getter(platform_name):
    original_platform = K.platform.name
    K.set_platform(platform_name)
    assert K.platform.name == platform_name
    assert K.get_platform() == platform_name
    K.set_platform("test")
    K.set_platform(original_platform)


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
def test_cast(platform, array_type):
    target = np.random.random(10)
    final = K.to_numpy(K.cast(target, dtype=array_type))
    K.assert_allclose(final, target)


@pytest.mark.parametrize("array_type", [None, "float32", "float64"])
@pytest.mark.parametrize("format", ["coo", "csr", "csc", "dia"])
def test_sparse_cast(platform, array_type, format):
    from scipy import sparse
    sptarget = sparse.rand(512, 512, dtype=array_type, format=format)
    assert K.issparse(sptarget)
    final = K.to_numpy(K.cast(sptarget))
    target = sptarget.toarray()
    K.assert_allclose(final, target)
    if K.platform.name != "numba":  # pragma: no cover
        sptarget = getattr(K.sparse, sptarget.__class__.__name__)(sptarget)
        assert K.issparse(sptarget)
        final = K.to_numpy(K.cast(sptarget))
        K.assert_allclose(final, target)


def test_to_numpy(platform):
    x = [0, 1, 2]
    target = K.to_numpy(K.cast(x))
    if K.platform.name == "numba":
        final = K.to_numpy(x)
    else: # pragma: no cover
        final = K.to_numpy(np.array(x))
    K.assert_allclose(final, target)


def test_basic_matrices(platform):
    K.assert_allclose(K.eye(4), np.eye(4))
    K.assert_allclose(K.zeros((3, 3)), np.zeros((3, 3)))
    K.assert_allclose(K.ones((5, 5)), np.ones((5, 5)))
    from scipy.linalg import expm
    m = np.random.random((4, 4))
    K.assert_allclose(K.expm(m), expm(m))


@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_backend_eigh(platform, sparse_type):
    if sparse_type is None:
        m = np.random.random((16, 16))
        eigvals1, eigvecs1 = K.eigh(K.cast(m))
        eigvals2, eigvecs2 = np.linalg.eigh(m)
    else:
        from scipy.sparse import rand
        m = rand(16, 16, format=sparse_type)
        m = m + m.T
        eigvals1, eigvecs1 = K.eigh(K.cast(m), k=16)
        eigvals2, eigvecs2 = K.eigh(K.cast(m.toarray()))
    K.assert_allclose(eigvals1, eigvals2, atol=1e-10)
    K.assert_allclose(K.abs(eigvecs1), np.abs(eigvecs2), atol=1e-10)


def test_backend_eigvalsh(platform):
    m = np.random.random((16, 16))
    target = np.linalg.eigvalsh(m)
    result = K.eigvalsh(K.cast(m))
    K.assert_allclose(target, result)


@pytest.mark.parametrize("sparse_type", ["coo", "csr", "csc", "dia"])
@pytest.mark.parametrize("k", [6, 10])
def test_backend_eigh_sparse(platform, sparse_type, k):
    if K.get_platform() != "numba":
        pytest.skip("Skipping sparse eigenvalue test for GPU platforms "
                    "because it is unstable.")
    from scipy.sparse.linalg import eigsh
    from scipy import sparse
    from qibo import hamiltonians
    ham = hamiltonians.TFIM(6, h=1.0)
    m = getattr(sparse, f"{sparse_type}_matrix")(K.to_numpy(ham.matrix))
    eigvals1, eigvecs1 = K.eigh(K.cast(m), k)
    eigvals2, eigvecs2 = eigsh(m, k)
    eigvals1 = np.abs(K.to_numpy(eigvals1))
    eigvals2 = np.abs(K.to_numpy(eigvals2))
    K.assert_allclose(sorted(eigvals1), sorted(eigvals2))


def test_unique_and_gather(platform):
    samples = np.random.randint(0, 2, size=(100,))
    K.assert_allclose(K.unique(samples), np.unique(samples))
    indices = [0, 2, 6, 40]
    final = K.gather(samples, indices)
    K.assert_allclose(final, samples[indices])


def test_with_device(platform):
    with K.device("/CPU:0"):
        pass
    with K.device("/GPU:0"):
        pass
    with K.on_cpu():
        pass
    target = np.random.random(5)
    final = K.cpu_cast(target)
    K.assert_allclose(final, target)


def test_cpu_ops(platform):
    with K.on_cpu():
        pass


def test_cpu_fallback(platform):
    state = K.cpu_fallback(K.initial_state, 4)
