"""Tests for the ``quantum_info`` custom operators."""

import numba
import numpy as np
import pytest
from qibo.quantum_info import random_gaussian_matrix, random_quantum_channel


@pytest.fixture
def nthreads(backend):
    """Run the test with all the threads available to ``numba``.

    Thread-dependent behavior is only exercised when the JIT is enabled, e.g. by running
    ``NUMBA_DISABLE_JIT=0 pytest``, since the test configuration disables it by default.
    """
    if backend.platform != "numba":
        pytest.skip("Number of threads only matters for the ``numba`` backend.")

    previous = backend.nthreads
    backend.set_threads(numba.config.NUMBA_NUM_THREADS)
    yield backend.nthreads
    backend.set_threads(previous)


def test_random_density_matrix_bures_purity(backend):
    if not hasattr(backend.qinfo, "_random_density_matrix_bures"):
        pytest.skip(
            "Bures sampling is not implemented in the ``qinfo`` of this backend."
        )

    # average purity of qubit states sampled from the Bures measure
    # is (5 * N^2 + 1) / (2 * N * (N^2 + 2)) = 7 / 8, with N = 2, while
    # it is 4 / 5 for the Hilbert-Schmidt measure (which is what we get
    # if the unitary of the construction is not Haar-distributed)
    nsamples = 1000
    purities = []
    for seed in range(nsamples):
        backend.set_seed(seed)
        state = backend.qinfo._random_density_matrix_bures(2, 2, 0.0, 1.0)
        purities.append(backend.real(backend.trace(state @ state)))

    backend.assert_allclose(backend.mean(purities), 7 / 8, atol=2e-2)


def test_random_gaussian_matrix_seed(backend, nthreads):
    matrices = [random_gaussian_matrix(8, 3, seed=7, backend=backend) for _ in range(3)]

    for matrix in matrices[1:]:
        backend.assert_allclose(matrix, matrices[0])


def test_random_quantum_channel_bcsz_seed(backend, nthreads):
    channels = [
        random_quantum_channel(
            4,
            measure="bcsz",
            rank=2,
            representation="choi",
            seed=7,
            backend=backend,
        )
        for _ in range(3)
    ]

    for channel in channels[1:]:
        backend.assert_allclose(channel, channels[0])

    # ``assert_allclose`` treats NaNs as equal, hence the channel is also checked to be
    # valid: finite and trace-preserving (partial trace over the output is the identity)
    assert np.all(np.isfinite(channels[0]))
    traced = backend.einsum("ijik->jk", backend.reshape(channels[0], (4, 4, 4, 4)))
    backend.assert_allclose(traced, backend.identity(4), atol=1e-8)
