"""Memory-bounded cache that reuses deterministic adapter outputs across candidates."""
from __future__ import annotations

from collections import OrderedDict
import copy
import json
import sys
import time

import numpy as np


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError('recipes must contain JSON-compatible values')


def recipe_key(recipe):
    return json.dumps(recipe, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False, default=_json_value)


def _bytes(value, seen=None):
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        size = int(sys.getsizeof(value) + (0 if value.base is None else value.nbytes))
        if value.dtype.hasobject:
            size += sum(_bytes(item, seen) for item in value.flat)
        return size
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(_bytes(k, seen) + _bytes(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(_bytes(v, seen) for v in value)
    return sys.getsizeof(value)


def _description(recipe):
    if recipe is None:
        return 'default'
    if isinstance(recipe, dict):
        return 'keys=' + ','.join(sorted(str(k) for k in recipe))
    return 'type=' + type(recipe).__name__


class AdapterOutputCache:
    def __init__(self, load, max_bytes=256 * 1024 * 1024):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError('max_bytes must be a positive integer')
        self.load = load
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._stats = dict(raw_calls=0, cache_hits=0, raw_load_seconds=0.0,
                           failures=0, evictions=0, bytes=0, entries=0)

    @property
    def cache_stats(self):
        return dict(self._stats)

    @property
    def aggregate_cache_stats(self):
        return getattr(self.load, 'aggregate_cache_stats', None)

    def __call__(self, recipe=None):
        key = recipe_key(recipe)
        if key in self._entries:
            value, size = self._entries.pop(key)
            self._entries[key] = (value, size)
            self._stats['cache_hits'] += 1
            print('  [adapter-cache] hit %s raw_calls=%d cache_hits=%d raw_load_seconds=%.3f' %
                  (_description(recipe), self._stats['raw_calls'], self._stats['cache_hits'],
                   self._stats['raw_load_seconds']), flush=True)
            return copy.deepcopy(value)
        description = _description(recipe)
        print('  [adapter-cache] load start ' + description, flush=True)
        started = time.perf_counter()
        self._stats['raw_calls'] += 1
        try:
            value = self.load(recipe)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._stats['raw_load_seconds'] = round(self._stats['raw_load_seconds'] + elapsed, 6)
            self._stats['failures'] += 1
            print('  [adapter-cache] load error %s type=%s raw_calls=%d failures=%d '
                  'raw_load_seconds=%.3f' %
                  (description, type(exc).__name__, self._stats['raw_calls'],
                   self._stats['failures'], self._stats['raw_load_seconds']), flush=True)
            raise
        elapsed = time.perf_counter() - started
        self._stats['raw_load_seconds'] = round(self._stats['raw_load_seconds'] + elapsed, 6)
        stored = copy.deepcopy(value)
        size = _bytes(stored)
        if size <= self.max_bytes:
            while self._entries and self._bytes + size > self.max_bytes:
                _, (_, removed) = self._entries.popitem(last=False)
                self._bytes -= removed
                self._stats['evictions'] += 1
            self._entries[key] = (stored, size)
            self._bytes += size
        self._stats['bytes'] = self._bytes
        self._stats['entries'] = len(self._entries)
        print('  [adapter-cache] load end %s %.3fs bytes=%d cached=%s raw_calls=%d '
              'cache_hits=%d raw_load_seconds=%.3f' %
              (description, elapsed, size, size <= self.max_bytes, self._stats['raw_calls'],
               self._stats['cache_hits'], self._stats['raw_load_seconds']), flush=True)
        return copy.deepcopy(value)
