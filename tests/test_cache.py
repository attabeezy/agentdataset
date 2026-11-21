import tempfile
from pathlib import Path

from agentdataset.core.cache import ContentCache


def test_cache_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        c = ContentCache(base_dir=str(Path(d) / "cache"))
        key = c.key("a", 1, "b")
        assert not c.has(key)
        c.set(key, {"x": 5})
        assert c.has(key)
        assert c.get(key) == {"x": 5}
        c.invalidate(key)
        assert c.get(key) is None


def test_cache_key_distinct():
    with tempfile.TemporaryDirectory() as d:
        c = ContentCache(base_dir=str(Path(d) / "cache"))
        assert c.key("a") != c.key("b")
        assert c.key("a", "b") != c.key("b", "a")