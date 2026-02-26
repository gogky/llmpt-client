"""
Pytest configuration and fixtures for integration tests.
"""

import os
import pytest
import shutil
import tempfile
import llmpt

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration", action="store_true", default=False, help="run integration tests"
    )

def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        # --run-integration given in cli: do not skip integration tests
        return
        
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip_integration)


@pytest.fixture
def live_tracker_url():
    """Return the URL of the live test tracker."""
    return "http://118.195.159.242"


@pytest.fixture
def clean_hf_cache():
    """Provides a clean temporary directory for HF cache."""
    # Store original cache dir
    orig_cache = os.environ.get("HF_HOME")
    
    # Create temp cache
    temp_dir = tempfile.mkdtemp(prefix="llmpt_integration_hf_")
    os.environ["HF_HOME"] = temp_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(temp_dir, "hub")
    
    yield
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)
    if orig_cache:
        os.environ["HF_HOME"] = orig_cache
        os.environ["HF_HUB_CACHE"] = os.path.join(orig_cache, "hub")
    else:
        del os.environ["HF_HOME"]
        del os.environ["HF_HUB_CACHE"]


@pytest.fixture(autouse=True)
def ensure_llmpt_cleanup():
    """Ensure llmpt is disabled and patches are removed between tests."""
    yield
    llmpt.disable_p2p()
    # Attempt to gracefully shutdown the batch manager if initialized
    from llmpt.p2p_batch import P2PBatchManager
    if P2PBatchManager._instance:
        # Give dict a copy to avoid mutation errors during shutdown
        for ctx in list(P2PBatchManager._instance.sessions.values()):
            ctx.is_valid = False
            if ctx.handle:
                ctx.lt_session.remove_torrent(ctx.handle)
        P2PBatchManager._instance = None
