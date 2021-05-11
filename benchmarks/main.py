import os
import argparse
import json
import time
import numpy as np
from backends import backends


parser = argparse.ArgumentParser()
parser.add_argument("--nqubits", default=20, type=int)
parser.add_argument("--backend", default="numba", type=str)
parser.add_argument("--method", default="qft", type=str)
parser.add_argument("--controls", default="", type=str)
parser.add_argument("--random", action="store_true")
parser.add_argument("--filename", default=None, type=str)


def random_state(nqubits, dtype="complex128"):
    x = np.random.random(2 ** nqubits) + 1j * np.random.random(2 ** nqubits)
    return (x / np.sqrt(np.sum(np.abs(x) ** 2))).astype(dtype)


def main(nqubits, backend, method, controls, random, filename):
    if filename is not None:
        if os.path.isfile(filename):
            with open(filename, "r") as file:
                logs = json.load(file)
            print("Extending existing logs from {}.".format(filename))
        else:
            print("Creating new logs in {}.".format(filename))
            logs = []
    else:
        logs = []

    logs.append({
        "nqubits": nqubits, "backend": backend, "random": random,
        "method": method, "controls": controls,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")
        })

    backend = backends.get(backend)
    if random:
        state = random_state(nqubits)
    else:
        state = np.zeros(2 ** nqubits, dtype=np.complex128)
        state[0] = 1

    start_time = time.time()
    state = backend.cast(state)
    logs[-1]["cast_time"] = time.time() - start_time

    controls = [int(q) for q in controls.split(",") if q]
    start_time = time.time()
    args = getattr(backend, "{}_args".format(method))(state, nqubits, controls)
    logs[-1]["prepare_time"] = time.time() - start_time

    start_time = time.time()
    state = getattr(backend, method)(*args)
    logs[-1]["dry_run_time"] = time.time() - start_time
    start_time = time.time()
    host_state = backend.to_numpy(state)
    logs[-1]["dry_run_to_host_time"] = time.time() - start_time
    logs[-1]["dry_run_state"] = repr(host_state)
    print("Dry run final state:")
    print(logs[-1]["dry_run_state"])
    print()

    start_time = time.time()
    state = getattr(backend, method)(*args)
    logs[-1]["execution_time"] = time.time() - start_time
    start_time = time.time()
    host_state = backend.to_numpy(state)
    logs[-1]["execution_to_host_time"] = time.time() - start_time
    logs[-1]["second_run_state"] = repr(host_state)
    print("Second run final state:")
    print(logs[-1]["second_run_state"])
    print()

    for k, v in logs[-1].items():
        print("{}: {}".format(k, v))
    print()

    if filename is not None:
        with open(filename, "w") as file:
            json.dump(logs, file)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
