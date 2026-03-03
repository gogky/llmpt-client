"""
Tests for revision resolution (resolve_commit_hash).
"""

import threading
import pytest
from unittest.mock import patch, MagicMock

from llmpt.utils import resolve_commit_hash, _revision_cache, _revision_cache_lock, _COMMIT_HASH_RE


@pytest.fixture(autouse=True)
def clear_revision_cache():
    """Clear the revision cache before and after each test."""
    with _revision_cache_lock:
        _revision_cache.clear()
    yield
    with _revision_cache_lock:
        _revision_cache.clear()


class TestCommitHashRegex:
    """Test the commit hash regex pattern."""

    def test_valid_40_char_hex(self):
        assert _COMMIT_HASH_RE.match("a" * 40) is not None
        assert _COMMIT_HASH_RE.match("0123456789abcdef" * 2 + "01234567") is not None

    def test_realistic_commit_hash(self):
        assert _COMMIT_HASH_RE.match("e7da7f221d5bf496a48136c0cd264e630fe9fcc8") is not None

    def test_uppercase_rejected(self):
        assert _COMMIT_HASH_RE.match("A" * 40) is None

    def test_too_short(self):
        assert _COMMIT_HASH_RE.match("abc123") is None

    def test_too_long(self):
        assert _COMMIT_HASH_RE.match("a" * 41) is None

    def test_branch_names_rejected(self):
        assert _COMMIT_HASH_RE.match("main") is None
        assert _COMMIT_HASH_RE.match("v1.0") is None
        assert _COMMIT_HASH_RE.match("refs/heads/main") is None

    def test_non_hex_chars_rejected(self):
        assert _COMMIT_HASH_RE.match("g" * 40) is None
        assert _COMMIT_HASH_RE.match("z" * 40) is None


class TestResolveCommitHash:
    """Test the resolve_commit_hash function."""

    FAKE_HASH = "e7da7f221d5bf496a48136c0cd264e630fe9fcc8"

    def test_already_commit_hash_returns_immediately(self):
        """If revision is already a 40-char hex hash, it should be returned as-is
        without any API call."""
        result = resolve_commit_hash("gpt2", self.FAKE_HASH)
        assert result == self.FAKE_HASH

    @patch("huggingface_hub.HfApi")
    def test_branch_name_resolved(self, MockHfApi):

        """Branch name 'main' should be resolved via HfApi.repo_info()."""
        mock_api = MockHfApi.return_value
        mock_info = MagicMock()
        mock_info.sha = self.FAKE_HASH
        mock_api.repo_info.return_value = mock_info

        result = resolve_commit_hash("gpt2", "main")
        assert result == self.FAKE_HASH

        mock_api.repo_info.assert_called_once_with(
            repo_id="gpt2", revision="main", repo_type="model"
        )

    @patch("huggingface_hub.HfApi")
    def test_result_is_cached(self, MockHfApi):
        """Second call with same args should use cache (no API call)."""
        mock_api = MockHfApi.return_value
        mock_info = MagicMock()
        mock_info.sha = self.FAKE_HASH
        mock_api.repo_info.return_value = mock_info

        # First call: API is called
        result1 = resolve_commit_hash("gpt2", "main")
        assert result1 == self.FAKE_HASH
        assert mock_api.repo_info.call_count == 1

        # Second call: cached, no API call
        result2 = resolve_commit_hash("gpt2", "main")
        assert result2 == self.FAKE_HASH
        assert mock_api.repo_info.call_count == 1  # still 1

    @patch("huggingface_hub.HfApi")
    def test_different_repo_not_cached(self, MockHfApi):
        """Different repo_id should trigger a fresh API call."""
        mock_api = MockHfApi.return_value
        mock_info = MagicMock()
        mock_info.sha = self.FAKE_HASH
        mock_api.repo_info.return_value = mock_info

        resolve_commit_hash("gpt2", "main")
        resolve_commit_hash("bert-base", "main")
        assert mock_api.repo_info.call_count == 2

    @patch("huggingface_hub.HfApi")
    def test_repo_type_passed_through(self, MockHfApi):
        """repo_type should be forwarded to HfApi.repo_info()."""
        mock_api = MockHfApi.return_value
        mock_info = MagicMock()
        mock_info.sha = self.FAKE_HASH
        mock_api.repo_info.return_value = mock_info

        resolve_commit_hash("some/dataset", "v1.0", repo_type="dataset")
        mock_api.repo_info.assert_called_once_with(
            repo_id="some/dataset", revision="v1.0", repo_type="dataset"
        )

    @patch("huggingface_hub.HfApi")
    def test_api_error_propagates(self, MockHfApi):
        """If HfApi.repo_info() fails, the exception should propagate."""
        mock_api = MockHfApi.return_value
        mock_api.repo_info.side_effect = Exception("Network error")

        with pytest.raises(Exception, match="Network error"):
            resolve_commit_hash("gpt2", "main")

    @patch("huggingface_hub.HfApi")
    def test_invalid_sha_from_api_raises(self, MockHfApi):
        """If the API returns an invalid sha, a ValueError should be raised."""
        mock_api = MockHfApi.return_value
        mock_info = MagicMock()
        mock_info.sha = "not-a-hash"
        mock_api.repo_info.return_value = mock_info

        with pytest.raises(ValueError, match="unexpected sha"):
            resolve_commit_hash("gpt2", "main")

    @patch("huggingface_hub.HfApi")
    def test_none_sha_from_api_raises(self, MockHfApi):
        """If the API returns None sha, a ValueError should be raised."""
        mock_api = MockHfApi.return_value
        mock_info = MagicMock()
        mock_info.sha = None
        mock_api.repo_info.return_value = mock_info

        with pytest.raises(ValueError, match="unexpected sha"):
            resolve_commit_hash("gpt2", "main")

    @patch("huggingface_hub.HfApi")
    def test_thread_safety(self, MockHfApi):
        """Multiple threads resolving the same revision should all get the same result
        and the API should be called at most once (or a few times due to race conditions)."""
        mock_api = MockHfApi.return_value
        mock_info = MagicMock()
        mock_info.sha = self.FAKE_HASH
        mock_api.repo_info.return_value = mock_info

        results = []
        errors = []

        def resolve():
            try:
                result = resolve_commit_hash("gpt2", "main")
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r == self.FAKE_HASH for r in results)
        assert len(results) == 10
