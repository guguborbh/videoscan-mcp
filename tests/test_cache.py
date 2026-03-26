"""Tests for disk cache with SQLite index."""
import json
from pathlib import Path
import pytest
from videoscan.core.cache import Cache

@pytest.fixture
def cache(tmp_cache_dir: Path) -> Cache:
    return Cache(cache_dir=str(tmp_cache_dir), max_size_gb=1)

def test_store_and_retrieve_result(cache):
    data = {"metadata": {"title": "Test"}, "frames": []}
    cache.store_result("vid123", "params_abc", data)
    result = cache.get_result("vid123", "params_abc")
    assert result is not None
    assert result["metadata"]["title"] == "Test"

def test_cache_miss_returns_none(cache):
    assert cache.get_result("nonexistent", "params") is None

def test_cache_expiry(cache):
    cache.store_result("vid123", "params", {"data": True}, ttl=0)
    assert cache.get_result("vid123", "params") is None

def test_force_refresh_bypasses_cache(cache):
    cache.store_result("vid123", "params", {"data": True})
    assert cache.get_result("vid123", "params", force_refresh=True) is None

def test_store_and_retrieve_frames(cache):
    cache.store_frames("vid123", [b"frame1_bytes", b"frame2_bytes"])
    frames = cache.get_frames("vid123")
    assert frames is not None
    assert len(frames) == 2

def test_cleanup_expired(cache):
    cache.store_result("old", "params", {"data": True}, ttl=0)
    cache.store_result("new", "params", {"data": True}, ttl=999999)
    cache.cleanup()
    assert cache.get_result("old", "params") is None
    assert cache.get_result("new", "params") is not None

def test_disabled_cache():
    cache = Cache(cache_dir="/tmp/disabled", enabled=False)
    cache.store_result("vid", "params", {"data": True})
    assert cache.get_result("vid", "params") is None
