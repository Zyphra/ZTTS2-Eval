"""Atomic file writes: write a temp file in the target's directory, then os.replace().

Readers (the explorer, plot scripts, a concurrent metric job) can never observe a
half-written file, and a crash leaves the previous version intact. The result keeps
the permissions of the file it replaces (a new file gets the umask default, as
open() would give it) rather than mkstemp's private 0600. Stdlib only.
"""
import json
import os
import tempfile


def _mode_for(path):
    try:
        return os.stat(path).st_mode & 0o777
    except OSError:
        umask = os.umask(0)
        os.umask(umask)
        return 0o666 & ~umask


def _write_atomic(path, write, **open_kw):
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    mode = _mode_for(path)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=os.path.basename(path) + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", **open_kw) as f:
            write(f)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text_atomic(path, text, encoding="utf-8"):
    """Write `text` to `path` atomically (creating the parent directory)."""
    _write_atomic(path, lambda f: f.write(text), encoding=encoding)


def write_json_atomic(path, obj, **json_kw):
    """json.dump `obj` to `path` atomically (creating the parent directory)."""
    _write_atomic(path, lambda f: json.dump(obj, f, **json_kw))
