"""Stage and progress lines for long runs."""
from __future__ import annotations

import os
import sys
import time

PROGRESS_EVERY_S = float(os.environ.get("MERID_PROGRESS_EVERY_S", "15"))
_T0 = time.monotonic()


def _fmt_s(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm%02ds" % divmod(seconds, 60)
    h, rest = divmod(seconds, 3600)
    return "%dh%02dm" % (h, rest // 60)


def _tty():
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def stage(name, traj=None, detail=""):
    bits = ["[%s]" % _fmt_s(time.monotonic() - _T0)]
    if traj is not None:
        s = traj.info.get("model_stats", {})
        bits.append("%d calls $%.3f" % (s.get("api_calls", 0), s.get("cost", 0.0)))
    line = " | ".join(bits) + "  " + "-" * 4 + " " + name
    if detail:
        line += "  (" + detail + ")"
    print(line, flush=True)


class Progress:
    def __init__(self, label, total, every_s=None):
        self.label, self.total = label, max(0, int(total or 0))
        self.n, self.t0 = 0, time.monotonic()
        self.every_s = PROGRESS_EVERY_S if every_s is None else every_s
        self._last_print, self._last_pct = -1e9, -1
        self._tty = _tty()
        self._note = ""

    def _line(self):
        el = time.monotonic() - self.t0
        pct = 100.0 * self.n / self.total if self.total else 0.0
        eta = (el / self.n * (self.total - self.n)) if (self.n and self.total) else None
        core = "  [%s] %d/%d (%.0f%%)  elapsed %s" % (self.label, self.n, self.total, pct, _fmt_s(el))
        if eta is not None:
            core += "  eta %s" % _fmt_s(eta)
        if self._note:
            core += "  " + self._note
        return core

    def update(self, n=1, note=None):
        self.n += n
        if note is not None:
            self._note = note
        now = time.monotonic()
        pct = int(10 * self.n / self.total) if self.total else 0
        if self.n >= self.total:
            return
        due = (now - self._last_print >= self.every_s) or pct != self._last_pct
        if not due:
            return
        self._last_print, self._last_pct = now, pct
        if self._tty and self.n < self.total:
            sys.stdout.write("\r" + self._line() + " " * 8)
            sys.stdout.flush()
        else:
            if self._tty:
                sys.stdout.write("\r")
            print(self._line(), flush=True)

    def done(self, note=None):
        if note is not None:
            self._note = note
        if self._tty:
            sys.stdout.write("\r")
        print(self._line() + "  done", flush=True)


class _Tee:
    def __init__(self, console, fh):
        self._c, self._f = console, fh

    def write(self, s):
        try:
            self._c.write(s)
        except Exception:
            pass
        if "\r" not in s:
            self._f.write(s)
        return len(s)

    def flush(self):
        for x in (self._c, self._f):
            try:
                x.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return self._c.isatty()
        except Exception:
            return False

    def fileno(self):
        return self._c.fileno()

    def __getattr__(self, name):
        return getattr(self._c, name)


def tee_output(path):
    if not hasattr(sys.stdout, "fileno"):
        return None
    try:
        sys.stdout.fileno()
    except Exception:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)
    for stream in ("stdout", "stderr"):
        try:
            getattr(sys, stream).reconfigure(line_buffering=True)
        except Exception:
            pass
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    print("[log] mirroring this run to %s" % path, flush=True)
    return path
