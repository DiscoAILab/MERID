"""LLM proposer (Anthropic Messages API) and the random-search baseline."""
from __future__ import annotations

import json
import os
import random
import re

from space import DEFAULT, SPACE, choices, roles, validate

_SYSTEM = ("You are an AutoML research agent constructing a multimodal depression-detection "
           "pipeline. You SEE the dataset's characteristics and pick the next configuration to "
           "MAXIMIZE the out-of-fold (OOF) track = (binaryMacroF1 + ternaryMacroF1 + PHQ_CCC)/3. "
           "Reason first about the DATA REGIME (is it small? p>>n? imbalanced?) and which paradigm "
           "fits, THEN choose concrete values.")

_JSON_RE = re.compile(r"\{[^{}]*\}")


def _space_desc() -> str:
    return "\n".join(f"  {ax} ({role}): {choices(ax)}" for ax, (role, _) in SPACE.items())


def _hist_line(h) -> str:
    if h.get("track") is None:
        return f"- {h['choice']} -> FAILED ({h.get('status', 'error')})"
    return (f"- {h['choice']} -> track={h['track']:.4f} "
            f"(binF1={h['binF1']:.3f}, terF1={h['terF1']:.3f}, ccc={h['ccc']:.3f})")


def _prompt(profile, history) -> str:
    hist = "\n".join(_hist_line(h) for h in history[-25:]) or "(none yet)"
    hints = "\n".join(f"  - {x}" for x in profile.get("hints", []))
    return (
        f"DATA REGIME (what you see):\n  {profile['summary']}\nStrategy hints:\n{hints}\n\n"
        f"Action space (pick exactly one value per axis; axis role in parens):\n{_space_desc()}\n\n"
        f"Configs already evaluated on OOF (do NOT repeat any):\n{hist}\n\n"
        "Given the regime, reason briefly about what to try next (e.g. which data recipe / "
        "representation / regularization the small-n p>>n setting favors, and how to lift the "
        "lagging PHQ_ccc without hurting F1). Then output ONLY a JSON object with keys "
        f"{list(SPACE)} as the FINAL line of your reply."
    )


def _parse(text: str, *, strict=False) -> dict:
    for cand in reversed(_JSON_RE.findall(text or "")):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and any(k in obj for k in SPACE):
            if strict and (set(obj) != set(SPACE) or
                           any(obj[axis] not in choices(axis) for axis in SPACE)):
                raise ValueError('LLM configuration must specify every bounded action-space axis')
            return validate(obj)
    if strict:
        raise ValueError('LLM response did not contain a valid configuration')
    print(f"  [parse] WARN no valid config JSON; snapping to DEFAULT. tail={(text or '')[-160:]!r}", flush=True)
    return dict(DEFAULT)


class RandomProposer:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.tried = set()
        self.last_reasoning = ""

    def propose(self, profile, history):
        choice = dict(DEFAULT)
        for _ in range(200):
            choice = {ax: self.rng.choice(choices(ax)) for ax in SPACE}
            k = tuple(choice[ax] for ax in SPACE)
            if k not in self.tried:
                self.tried.add(k)
                break
        return choice


class ClaudeProposer:
    def __init__(self, model="claude-opus-4-8", provider="anthropic", base_url=None, api_key=None,
                 temperature=None, strict_config=False):
        self.model = model
        self.provider = provider
        self.base_url = base_url
        # Sent only when set, since Claude Opus 4.8 does not accept a temperature parameter.
        self.temperature = temperature
        self.strict_config = strict_config
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self.last_reasoning = ""

    def _usd(self, input_tokens, output_tokens):
        return input_tokens / 1e6 * 5 + output_tokens / 1e6 * 25

    def _client_lazy(self):
        if self._client is None:
            import anthropic
            base = (self.base_url or os.environ.get("ANTHROPIC_BASE_URL")
                    or os.environ.get("ANTHROPIC_API_BASE"))
            self._client = (anthropic.Anthropic(api_key=self.api_key, base_url=base)
                            if base else anthropic.Anthropic(api_key=self.api_key))
        return self._client

    def propose(self, profile, history):
        kw = {} if self.temperature is None else {"temperature": self.temperature}
        resp = self._client_lazy().messages.create(
            model=self.model, max_tokens=4096, system=_SYSTEM,
            messages=[{"role": "user", "content": _prompt(profile, history)}], **kw,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        self.last_reasoning = text
        traj = getattr(self, "traj", None)
        if traj is not None:
            u = getattr(resp, "usage", None)
            it = getattr(u, "input_tokens", 0) or 0
            ot = getattr(u, "output_tokens", 0) or 0
            traj.record_call(it, ot, cost=self._usd(it, ot))
        if text and text.rstrip()[-1] not in "\"}]":
            print("  [propose] WARN response may be truncated mid-JSON (raise max_tokens)", flush=True)
        return _parse(text, strict=self.strict_config)

    def complete(self, system, prompt, max_tokens=3000):
        return self.chat(system, [{"role": "user", "content": prompt}], max_tokens)

    def chat(self, system, messages, max_tokens=3000):
        kw = {} if self.temperature is None else {"temperature": self.temperature}
        resp = self._client_lazy().messages.create(
            model=self.model, max_tokens=max_tokens, system=system, messages=messages, **kw,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        self.last_reasoning = text
        traj = getattr(self, "traj", None)
        if traj is not None:
            u = getattr(resp, "usage", None)
            it = getattr(u, "input_tokens", 0) or 0
            ot = getattr(u, "output_tokens", 0) or 0
            traj.record_call(it, ot, cost=self._usd(it, ot))
        return text
