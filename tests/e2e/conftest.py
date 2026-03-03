"""
Pytest configuration for Docker-based E2E tests.

These tests are designed to run inside Docker containers via:
    docker compose -f docker-compose.test.yml up --build

They should NOT be collected when running pytest directly on the host.
"""

import pytest
import llmpt


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False, help="run E2E tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return

    skip_e2e = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "e2e" in str(item.fspath):
            item.add_marker(skip_e2e)


@pytest.fixture(autouse=True)
def ensure_llmpt_cleanup():
    """Ensure llmpt is disabled and patches are removed between tests."""
    yield
    llmpt.disable_p2p()
    from llmpt.p2p_batch import P2PBatchManager
    if P2PBatchManager._instance:
        for ctx in list(P2PBatchManager._instance.sessions.values()):
            ctx.is_valid = False
            if ctx.handle:
                ctx.lt_session.remove_torrent(ctx.handle)
        P2PBatchManager._instance = None
