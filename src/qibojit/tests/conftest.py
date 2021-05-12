def pytest_generate_tests(metafunc):
    if "dtype" in metafunc.fixturenames:
        metafunc.parametrize("dtype", ["complex128", "complex64"])
