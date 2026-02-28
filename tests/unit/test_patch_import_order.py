"""
Test: apply_patch correctly patches _snapshot_download.hf_hub_download
regardless of the order in which huggingface_hub is imported.

This directly validates the README claim:
  "import order does not affect the patch; only enable_p2p() must be called
   before any download starts."

NOTE on huggingface_hub.__getattr__ (lazy-import):
  The top-level `huggingface_hub` package uses a lazy-import __getattr__ that
  re-fetches attributes from sub-modules on every access.  This means that
  `huggingface_hub.hf_hub_download = patched` is silently overwritten the next
  time __getattr__ runs, so we deliberately do NOT assert on the top-level name.
  The two places that actually matter are:
    - huggingface_hub.file_download.hf_hub_download   (used by direct callers)
    - huggingface_hub._snapshot_download.hf_hub_download  (used by snapshot_download)
"""

import pytest
from huggingface_hub import _snapshot_download  # intentional: imported BEFORE apply_patch
from huggingface_hub import file_download

from llmpt.patch import apply_patch, remove_patch


@pytest.fixture(autouse=True)
def cleanup():
    yield
    remove_patch()


def test_snapshot_download_module_is_patched_after_apply():
    """
    Even though we imported _snapshot_download before apply_patch(),
    the module-level reference must be replaced when apply_patch() runs.

    This is the critical import-order test: the patch targets the MODULE OBJECT's
    __dict__, which is the same object regardless of when the import happened.
    """
    original = _snapshot_download.hf_hub_download

    apply_patch({'tracker_url': 'http://test-tracker'})

    assert _snapshot_download.hf_hub_download is not original, (
        "_snapshot_download.hf_hub_download was NOT replaced by apply_patch(). "
        "Import order still matters — README warning must be kept!"
    )
    assert callable(_snapshot_download.hf_hub_download)


def test_file_download_module_is_patched_after_apply():
    """file_download.hf_hub_download (used by direct hf_hub_download callers) must be replaced."""
    original = file_download.hf_hub_download

    apply_patch({'tracker_url': 'http://test-tracker'})

    assert file_download.hf_hub_download is not original, (
        "file_download.hf_hub_download was NOT replaced."
    )
    assert callable(file_download.hf_hub_download)


def test_remove_patch_restores_both():
    """
    remove_patch() must restore both module-level references back to the originals.
    """
    original_snap = _snapshot_download.hf_hub_download
    original_fd   = file_download.hf_hub_download

    apply_patch({'tracker_url': 'http://test-tracker'})
    remove_patch()

    assert _snapshot_download.hf_hub_download is original_snap, \
        "_snapshot_download.hf_hub_download was not restored after remove_patch()"
    assert file_download.hf_hub_download is original_fd, \
        "file_download.hf_hub_download was not restored after remove_patch()"


def test_double_apply_is_idempotent():
    """Calling apply_patch twice must not double-wrap the function."""
    apply_patch({'tracker_url': 'http://test-tracker'})
    patched_once = _snapshot_download.hf_hub_download

    apply_patch({'tracker_url': 'http://test-tracker'})  # second call
    patched_twice = _snapshot_download.hf_hub_download

    assert patched_once is patched_twice, \
        "apply_patch() is not idempotent — double application wraps the function twice!"
