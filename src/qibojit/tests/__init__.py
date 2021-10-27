from qibojit.custom_operators.backends import NumbaBackend, CupyBackend

try:
    backend = CupyBackend()
except (RuntimeError, ImportError):
    backend = NumbaBackend()
