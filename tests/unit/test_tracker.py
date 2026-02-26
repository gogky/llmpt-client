"""
Tests for tracker client using responses to mock HTTP calls.
"""

import pytest
import responses
import requests
from llmpt.tracker_client import TrackerClient

@pytest.fixture
def tracker_client():
    return TrackerClient("http://tracker.example.com")


def test_tracker_client_init():
    """Test TrackerClient initialization."""
    client = TrackerClient("http://tracker.example.com")
    assert client.tracker_url == "http://tracker.example.com"
    assert client.timeout == 10


def test_tracker_url_normalization():
    """Test that trailing slash is removed."""
    client = TrackerClient("http://tracker.example.com/")
    assert client.tracker_url == "http://tracker.example.com"


@responses.activate
def test_get_torrent_info_success(tracker_client):
    """Test successful retrieval of torrent info."""
    responses.add(
        responses.GET,
        "http://tracker.example.com/api/v1/torrents",
        json={
            "data": [
                {
                    "info_hash": "12345",
                    "magnet_link": "magnet:?xt=urn:btih:12345",
                    "revision": "main"
                },
                {
                    "info_hash": "67890",
                    "magnet_link": "magnet:?xt=urn:btih:67890",
                    "revision": "old_branch"
                }
            ],
            "total": 2
        },
        status=200
    )
    
    # 1. Test getting latest (no revision specified) - should return first item
    result = tracker_client.get_torrent_info("meta-llama/Llama-2", "model.bin")
    assert result is not None
    assert result["info_hash"] == "12345"
    
    # 2. Test getting specific revision
    result_rev = tracker_client.get_torrent_info("meta-llama/Llama-2", "model.bin", revision="old_branch")
    assert result_rev is not None
    assert result_rev["info_hash"] == "67890"

    # 3. Test non-existent revision
    result_none = tracker_client.get_torrent_info("meta-llama/Llama-2", "model.bin", revision="nonexistent")
    assert result_none is None


@responses.activate
def test_get_torrent_info_not_found(tracker_client):
    """Test handling of 404 from Tracker."""
    responses.add(
        responses.GET,
        "http://tracker.example.com/api/v1/torrents",
        status=404
    )
    
    result = tracker_client.get_torrent_info("demo", "file")
    assert result is None


@responses.activate
def test_get_torrent_info_empty_data(tracker_client):
    """Test handling of empty data list from Tracker."""
    responses.add(
        responses.GET,
        "http://tracker.example.com/api/v1/torrents",
        json={"data": [], "total": 0},
        status=200
    )
    
    result = tracker_client.get_torrent_info("demo", "file")
    assert result is None


@responses.activate
def test_get_torrent_info_timeout(tracker_client):
    """Test handling of request timeout."""
    responses.add(
        responses.GET,
        "http://tracker.example.com/api/v1/torrents",
        body=requests.exceptions.ReadTimeout()
    )
    
    result = tracker_client.get_torrent_info("demo", "file")
    assert result is None


@responses.activate
def test_register_torrent_success(tracker_client):
    """Test successful torrent registration."""
    responses.add(
        responses.POST,
        "http://tracker.example.com/api/v1/publish",
        json={"success": True},
        status=200
    )
    
    success = tracker_client.register_torrent(
        repo_id="test",
        revision="main",
        repo_type="model",
        name="test model",
        info_hash="abc",
        total_size=100,
        file_count=1,
        magnet_link="magnet:?xt=abc",
        piece_length=1024
    )
    
    assert success is True
    assert len(responses.calls) == 1
    
    # Verify exact JSON payload sent
    import json
    request_body = json.loads(responses.calls[0].request.body)
    assert request_body["repo_id"] == "test"
    assert request_body["info_hash"] == "abc"


@responses.activate
def test_register_torrent_failure(tracker_client):
    """Test failing torrent registration."""
    responses.add(
        responses.POST,
        "http://tracker.example.com/api/v1/publish",
        status=500
    )
    
    success = tracker_client.register_torrent(
        repo_id="test", revision="main", repo_type="model",
        name="test model", info_hash="abc", total_size=100,
        file_count=1, magnet_link="magnet:?xt=abc", piece_length=1024
    )
    
    assert success is False
