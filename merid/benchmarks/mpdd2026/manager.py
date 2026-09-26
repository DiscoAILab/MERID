"""MPDD-2026 search loop: the proposer suggests configurations and the harness scores them."""
from __future__ import annotations

import json
import os
import random
import time
import traceback
from pathlib import Path

from merid.benchmarks.mpdd2026.profiler import profile as build_profile
from merid.benchmarks.mpdd2026.space import DEFAULT, choices, key as config_key
from merid.benchmarks.mpdd2026.space import SPACE


def _failure_location(exc):
    return [dict(module=Path(frame.filename).stem, function=frame.name, line=frame.lineno)
            for frame in traceback.extract_tb(exc.__traceback__)]


def _random_unused(seen, rng):
    for _ in range(500):
        c = {ax: rng.choice(choices(ax)) for ax in SPACE}
        if config_key(c) not in seen:
            return c
    return dict(DEFAULT)


def _propose_hardened(proposer, profile, history, seen, rng, base_source, retries=3):
    for attempt in range(retries):
        try:
            c = proposer.propose(profile, history)
            reasoning = getattr(proposer, "last_reasoning", "")
            if config_key(c) in seen:
                c = _random_unused(seen, rng)
                reasoning += "\n[manager] duplicate proposal; substituted a random unused config."
                return c, reasoning, "random_duplicate_substitution"
            return c, reasoning, base_source
        except Exception as exc:
            wait = 2 ** attempt
            print(f"  [propose] API error (try {attempt + 1}/{retries}): {type(exc).__name__}; "
                  f"backoff {wait}s", flush=True)
            time.sleep(wait)
    print("  [propose] persistent API failure -> random unused config", flush=True)
    return _random_unused(seen, rng), "[manager] API failure fallback: random unused config.", "fallback"


def _load_history(log_path):
    if not (log_path and os.path.exists(log_path)):
        return []
    hist = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                hist.append(json.loads(line))
    return hist


def optimize(harness, proposer, budget=12, seed=42, log_path=None, profile=None,
             search_mode='phq_only', fit_budget=720, reference_components=None,
             retrospective=0, review_checks=2, review_mode='roles', revision_reserve=0.25,
             traj=None, ctx='MPDD-2026', search_axes='all'):
    if search_mode == 'joint_heads':
        from merid.benchmarks.mpdd2026.joint_search import optimize_joint
        return optimize_joint(harness, proposer, budget=budget, seed=seed,
                              log_path=log_path, profile=profile, fit_budget=fit_budget,
                              reference_components=reference_components,
                              retrospective=retrospective, review_checks=review_checks,
                              review_mode=review_mode, revision_reserve=revision_reserve,
                              traj=traj, ctx=ctx, search_axes=search_axes)
    if search_mode != 'phq_only':
        raise ValueError('search_mode must be phq_only or joint_heads')
    if search_axes != 'all':
        raise ValueError('search_axes is implemented for joint_heads only')
    if reference_components is not None:
        raise ValueError('reference_components requires joint_heads')
    if retrospective:
        raise ValueError('retrospective review is implemented for joint_heads only')
    rng = random.Random(seed)
    profile = profile or build_profile(harness)
    print("== REGIME PROFILE ==\n" + profile["summary"], flush=True)
    for h in profile.get("hints", []):
        print("  hint:", h, flush=True)

    history = _load_history(log_path)
    seen = {tuple(config_key(h["choice"])) for h in history if "choice" in h}
    if history:
        print(f"== resumed {len(history)} evals from {log_path} ==", flush=True)

    base_source = "random" if type(proposer).__name__ == "RandomProposer" else "llm"
    logf = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        while len(history) < budget:
            step = len(history) + 1
            choice, reasoning, source = _propose_hardened(proposer, profile, history, seen, rng, base_source)
            seen.add(tuple(config_key(choice)))
            try:
                r = harness.evaluate(choice)
                rec = {"step": step, "choice": choice, "track": r["track"],
                       "binF1": r["binF1"], "terF1": r["terF1"], "ccc": r["ccc"],
                       "name": r["name"], "source": source, "reasoning": reasoning}
                rec["evaluation"] = {key: value for key, value in r.items()
                                     if key not in ("config",)}
                print(f"[{step}/{budget}] track={r['track']:.4f} "
                      f"(binF1={r['binF1']:.3f} terF1={r['terF1']:.3f} ccc={r['ccc']:.3f})  "
                      f"{choice['audio']}/{choice['video']}/{choice['mode']}/{choice['model']}"
                      f"/k{choice['k']}/s{choice['shrink']}  [{source}]", flush=True)
            except Exception as exc:
                rec = {"step": step, "choice": choice, "track": None,
                       "status": type(exc).__name__, "source": source, "reasoning": reasoning,
                       "failure_location": _failure_location(exc)}
                location = rec["failure_location"][-1] if rec["failure_location"] else None
                suffix = (f" at {location['module']}.{location['function']}:{location['line']}"
                          if location else "")
                print(f"[{step}/{budget}] FAILED {choice}: {type(exc).__name__}{suffix}", flush=True)
            history.append(rec)
            if logf:
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()
    finally:
        if logf:
            logf.close()

    scored = [h for h in history if h.get("track") is not None]
    best = max(scored, key=lambda h: h["track"]) if scored else None
    return history, best
