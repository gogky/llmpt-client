"""Tests for shared session identity helpers."""


def test_build_storage_identity_uses_default_hub_cache():
    from llmpt.session_identity import build_storage_identity
    from llmpt.utils import get_hf_hub_cache

    identity = build_storage_identity()

    assert identity.kind == "hub_cache"
    assert identity.root == get_hf_hub_cache()


def test_build_storage_identity_prefers_local_dir():
    from llmpt.session_identity import build_storage_identity

    identity = build_storage_identity(
        cache_dir="/tmp/cache",
        local_dir="/tmp/local",
    )

    assert identity.kind == "local_dir"
    assert identity.root == "/tmp/local"


def test_build_source_session_key_exposes_legacy_shape():
    from llmpt.session_identity import build_source_session_key

    key = build_source_session_key(
        "model",
        "demo",
        "main",
        cache_dir="/tmp/cache",
    )

    assert key.repo_type == "model"
    assert key.repo_id == "demo"
    assert key.revision == "main"
    assert key.storage_kind == "hub_cache"
    assert key.storage_root == "/tmp/cache"
    assert key.as_legacy_tuple() == (
        "model",
        "demo",
        "main",
        "hub_cache",
        "/tmp/cache",
    )


def test_build_fastresume_filename_matches_storage_scope():
    from llmpt.session_identity import build_fastresume_filename
    from llmpt.utils import get_hf_hub_cache

    default_name = build_fastresume_filename("test/repo", "a" * 40)
    explicit_default_name = build_fastresume_filename(
        "test/repo",
        "a" * 40,
        cache_dir=get_hf_hub_cache(),
    )
    cache_name = build_fastresume_filename(
        "test/repo",
        "a" * 40,
        cache_dir="/tmp/custom_cache",
    )
    local_name = build_fastresume_filename(
        "test/repo",
        "a" * 40,
        local_dir="/tmp/local_dir",
    )

    assert default_name == explicit_default_name
    assert default_name != cache_name
    assert cache_name != local_name
