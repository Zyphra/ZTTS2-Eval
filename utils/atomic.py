"""Atomic file writes: write a temp file in the target's directory, then os.replace().

Readers (the explorer, plot scripts, a concurrent metric job) can never observe a
half-written file, and a crash leaves the previous version intact. Stdlib only.
"""
import json
import os
import tempfile


def write_text_atomic(path, text, encoding="utf-8"):
    """Write `text` to `path` atomically (creating the parent directory)."""
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json_atomic(path, obj, **json_kw):
    """json.dump `obj` to `path` atomically (creating the parent directory)."""
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, **json_kw)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
