"""
Tests for tracker client.
"""

import pytest
from llmpt.tracker_client import TrackerClient


def test_tracker_client_init():
    """Test TrackerClient initialization."""
    client = TrackerClient("http://tracker.example.com")
    assert client.tracker_url == "http://tracker.example.com"
    assert client.timeout == 10


def test_tracker_url_normalization():
    """Test that trailing slash is removed."""
    client = TrackerClient("http://tracker.example.com/")
    assert client.tracker_url == "http://tracker.example.com"


# Note: More tests require a running tracker server
# These would be integration tests
