"""Tests for sg.filelock — file locking."""
from __future__ import annotations

import threading
import time

import pytest

from sg.filelock import file_lock, FileLockTimeout


class TestFileLock:
    def test_basic_lock_unlock(self, tmp_path):
        target = tmp_path / "data.json"
        target.write_text("{}")
        with file_lock(target):
            target.write_text('{"locked": true}')
        assert target.read_text() == '{"locked": true}'

    def test_lock_creates_sidecar(self, tmp_path):
        target = tmp_path / "data.json"
        target.write_text("{}")
        with file_lock(target):
            assert (tmp_path / "data.json.lock").exists()

    def test_concurrent_locks_serialize(self, tmp_path):
        """Two threads contending for the same lock execute serially."""
        target = tmp_path / "counter.txt"
        target.write_text("0")
        results = []

        def _increment(delay):
            with file_lock(target, timeout=5.0):
                val = int(target.read_text())
                time.sleep(delay)
                target.write_text(str(val + 1))
                results.append(val + 1)

        t1 = threading.Thread(target=_increment, args=(0.1,))
        t2 = threading.Thread(target=_increment, args=(0.1,))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Both should have incremented, resulting in 2
        assert target.read_text() == "2"
        assert sorted(results) == [1, 2]

    def test_timeout_raises(self, tmp_path):
        """Lock times out when another holder doesn't release."""
        target = tmp_path / "data.json"
        target.write_text("{}")
        held = threading.Event()
        release = threading.Event()

        def _hold():
            with file_lock(target, timeout=5):
                held.set()
                release.wait(timeout=5)

        t = threading.Thread(target=_hold)
        t.start()
        held.wait(timeout=5)

        with pytest.raises(FileLockTimeout):
            with file_lock(target, timeout=0.2):
                pass

        release.set()
        t.join(timeout=5)

    def test_lock_on_nonexistent_parent(self, tmp_path):
        target = tmp_path / "sub" / "deep" / "data.json"
        with file_lock(target):
            pass
        assert (tmp_path / "sub" / "deep" / "data.json.lock").exists()
