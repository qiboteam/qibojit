import numpy as np
import pytest
from qibo import set_device
from qibo.hamiltonians import TFIM
from scipy import sparse
from scipy.linalg import expm as scipy_expm
from scipy.sparse.linalg import expm as expm_sparse

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
    sptarget = sparse.rand(64, 64, dtype=array_type, format=format)
    assert backend.is_sparse(sptarget)
    final = backend.to_numpy(backend.cast(sptarget))
    target = sptarget.toarray()
    backend.assert_allclose(final, target)
    if backend.platform != "numba":
        sptarget = getattr(backend.cp_sparse, sptarget.__class__.__name__)(sptarget)
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


@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_backend_expm(backend, sparse_type):
    rng = np.random.default_rng(10)
    if sparse_type is None:
        matrix = rng.random((8, 8))
        target = scipy_expm(matrix)
    else:
        matrix = sparse.rand(16, 16, format=sparse_type, rng=rng)
        target = expm_sparse(matrix)

    result = backend.cast(matrix, dtype=backend.float64, copy=True)
    result = backend.matrix_exp(result)

    backend.assert_allclose(
        backend.to_numpy(result), backend.to_numpy(target), atol=1e-10
    )


@pytest.mark.parametrize("sparse_type", [None, "coo", "csr", "csc", "dia"])
def test_backend_eigh(backend, sparse_type):
    if sparse_type is None:
        m = np.random.random((16, 16))
        eigvals1, eigvecs1 = backend.eigenvectors(backend.cast(m, dtype=m.dtype))
        eigvals2, eigvecs2 = np.linalg.eigh(m)
    else:
        m = sparse.rand(16, 16, format=sparse_type)
        m = m + m.T
        m = backend.cast(m, dtype=m.dtype)
        eigvals1, eigvecs1 = backend.eigenvectors(m, k=16)
        eigvals2, eigvecs2 = backend.eigenvectors(m.toarray())
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
        m = backend.cast(m, dtype=m.dtype)
        result = backend.eigenvalues(m, k=16)
        target, _ = backend.eigenvectors(m.toarray())
    backend.assert_allclose(result, target, atol=1e-10)


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


@pytest.mark.parametrize("size", [2, 3])
def test_backend_poly_matrix(backend, size):
    backend.set_seed(10)
    matrix = _random_complex((size, size), backend)

    target = np.poly(backend.to_numpy(matrix))
    result = backend.poly(matrix)

    backend.assert_allclose(
        result, backend.cast(target, dtype=result.dtype), atol=1e-10
    )


@pytest.mark.parametrize("degree", [1, 2, 5])
def test_backend_poly_roots(backend, degree):
    backend.set_seed(10)
    roots = _random_complex(degree, backend)

    target = np.poly(backend.to_numpy(roots))
    coefficients = backend.poly(roots)
    backend.assert_allclose(
        coefficients, backend.cast(target, dtype=coefficients.dtype), atol=1e-10
    )

    # the roots of the (monic) polynomial give back its coefficients, whatever their order
    backend.assert_allclose(
        backend.poly(backend.roots(coefficients)), coefficients, atol=1e-10
    )


def test_backend_roots_edge_cases(backend):
    # 2 * x^2 - 3 * x + 1, with two zeros in the highest degrees
    leading = backend.cast([0.0, 0.0, 2.0, -3.0, 1.0], dtype="float64")
    coefficients = backend.poly(backend.roots(leading))
    backend.assert_allclose(
        coefficients,
        backend.cast([1.0, -1.5, 0.5], dtype=coefficients.dtype),
        atol=1e-10,
    )

    # degree one
    linear = backend.cast([2.0, -6.0], dtype="float64")
    result = backend.roots(linear)
    backend.assert_allclose(result, backend.cast([3.0], dtype=result.dtype))

    # no roots
    constant = backend.cast([5.0], dtype="float64")
    assert len(backend.roots(constant)) == 0


def test_create_dtype(backend):
    assert backend.create_dtype("uint8") == np.dtype("uint8")
    assert backend.create_dtype("V3").itemsize == 3


def test_frombuffer(backend):
    array = backend.frombuffer(bytes([1, 2, 3]), dtype=backend.create_dtype("uint8"))

    np.testing.assert_array_equal(backend.to_numpy(array), np.array([1, 2, 3]))


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_packbits(backend, axis):
    array = np.random.default_rng(42).integers(0, 2, size=(11, 13))
    target = np.packbits(array, axis=axis)

    array = backend.cast(array, dtype=array.dtype)
    result = backend.packbits(array, axis=axis)

    np.testing.assert_array_equal(backend.to_numpy(result), target)


def test_metabackend_list_available():
    available_backends = {
        backend: backend in AVAILABLE_BACKENDS for backend in BACKENDS
    }
    assert MetaBackend().list_available() == available_backends


@pytest.mark.parametrize(
    "a1_dtype, a2, indices",
    [
        ("float64", np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2])),
        ("float64", np.array([1 + 1j, 2 + 2j, 3 + 3j]), np.array([0, 1, 2])),
        ("complex128", np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2])),
        ("complex128", np.array([1 + 1j, 2 + 2j, 3 + 3j]), np.array([0, 1, 2])),
    ],
)
def test_add_at_functionality(backend, a1_dtype, a2, indices):
    if a1_dtype == "float64":
        dtype = backend.engine.float64
        dtype_np = np.float64
    elif a1_dtype == "complex128":
        dtype = backend.engine.complex128
        dtype_np = np.complex128

    a1 = np.ones(5, dtype=dtype_np)

    a1_bkd = backend.cast(a1, dtype=dtype)
    a2_bkd = backend.cast(a2, dtype=a2.dtype)
    indices_bkd = backend.cast(indices, dtype=indices.dtype)

    expected = a1.copy()
    np.add.at(expected, indices, a2)

    actual = a1_bkd.copy()
    backend.add_at(actual, indices_bkd, a2_bkd)

    backend.assert_allclose(actual, backend.cast(expected, dtype=expected.dtype))


@pytest.mark.parametrize(
    "indices, a2, error",
    [
        (np.array([10]), np.array([1.0]), IndexError),
        (np.array([-11]), np.array([1.0]), IndexError),
        (np.array([0.1, 1.2]), np.array([1.0, 2.0]), IndexError),
        (np.array([0, 1, 2]), np.array([1.0, 2.0]), ValueError),
    ],
)
def test_add_at_errors(backend, indices, a2, error):
    a1 = backend.zeros(5)
    a2 = backend.cast(a2, dtype=a2.dtype)
    indices = backend.cast(indices, dtype=indices.dtype)
    with pytest.raises(error):
        backend.add_at(a1, indices, a2)


def _random_complex(size, backend):
    """Complex array of normally distributed numbers with the given shape."""
    real = backend.random_normal(0.0, 1.0, size=size, dtype="float64")
    imag = backend.random_normal(0.0, 1.0, size=size, dtype="float64")

    return backend.cast(real, dtype="complex128") + 1j * backend.cast(
        imag, dtype="complex128"
    )
