import qibo.quantum_info.quantum_info as qinfo
from numba import njit
from qibo import parallel

SIGNATURES = {
    "_pauli_basis": "c16[:,:,:](i8, c16[:,:], c16[:,:], c16[:,:], c16[:,:], f8)",
}


class QinfoNumba:
    pass


QINFO = QinfoNumba()

for function, signature in SIGNATURES.items():
    jitted = njit(signature=signature, parallel=True, cache=True)(
        getattr(qinfo, function)
    )
    setattr(QINFO, function, jitted)
