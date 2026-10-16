def pytest_addoption(parser):
    parser.addoption("--data-config", default=None, help="Explicit config for read-only external-data integration")
