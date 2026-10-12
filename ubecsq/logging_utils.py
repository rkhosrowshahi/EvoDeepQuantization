"""Mirror the console output of a run into a log file."""

import contextlib
import os
import sys


class _Tee:
    """Mirrors writes to several streams. isatty() follows the first (real) stream,
    so tqdm's terminal detection is unaffected."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return self.streams[0].isatty()


@contextlib.contextmanager
def tee_stdio_to_file(log_path):
    """Copy stdout and stderr into `log_path` (truncated) for the duration of the block."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, log_file)
    sys.stderr = _Tee(orig_err, log_file)
    try:
        yield log_path
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        log_file.close()
