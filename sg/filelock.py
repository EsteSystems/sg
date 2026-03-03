"""File locking — fcntl.flock() on Unix, directory-creation fallback elsewhere.

Used by Registry.save_index() and PhenotypeMap.save() to prevent
concurrent writes from corrupting JSON/TOML state files.

Usage::

    from sg.filelock import file_lock

    with file_lock(path):
        path.write_text(data)
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


from sg.log import get_logger

logger = get_logger("filelock")


class FileLockTimeout(Exception):
    """Raised when a file lock cannot be acquired within the timeout."""


def atomic_write_text(path: Path, content: str) -> None:
    """Write content to path atomically via temp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(str(tmp), str(path))


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Write bytes to path atomically via temp file + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(content)
    os.replace(str(tmp), str(path))


@contextmanager
def file_lock(path: Path, timeout: float = 10.0) -> Generator[None, None, None]:
    """Acquire an exclusive lock on *path* for the duration of the block.

    On Unix, uses ``fcntl.flock()`` on a ``.lock`` sidecar file.
    On other platforms, falls back to a lock-directory strategy.

    Raises :class:`FileLockTimeout` if the lock cannot be acquired within
    *timeout* seconds.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl
        yield from _flock_unix(lock_path, timeout, fcntl)
    except ImportError:
        yield from _flock_directory(lock_path, timeout)


def _flock_unix(
    lock_path: Path, timeout: float, fcntl
) -> Generator[None, None, None]:
    """fcntl.flock() implementation (Unix)."""
    deadline = time.monotonic() + timeout
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (IOError, OSError):
                if time.monotonic() >= deadline:
                    raise FileLockTimeout(
                        f"could not acquire lock on {lock_path} "
                        f"within {timeout}s"
                    )
                time.sleep(0.05)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _flock_directory(
    lock_path: Path, timeout: float,
) -> Generator[None, None, None]:
    """Directory-creation fallback (non-Unix)."""
    dir_lock = lock_path.with_suffix(".dir")
    deadline = time.monotonic() + timeout

    while True:
        try:
            dir_lock.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise FileLockTimeout(
                    f"could not acquire lock on {dir_lock} "
                    f"within {timeout}s"
                )
            time.sleep(0.05)

    try:
        yield
    finally:
        try:
            dir_lock.rmdir()
        except OSError:
            pass
