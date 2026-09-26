"""Extends the method catalog: generates new estimator code, smoke-tests it and validates it on the development metric."""
from __future__ import annotations

import re

import numpy as np

MODULE_SYSTEM = (
    "You are an ML engineer writing a small model component for a SMALL-SAMPLE (n~100), "
    "p>>n multimodal depression-detection pipeline. Given the data regime, write ONE Python "
    "function `build()` that returns a scikit-learn-compatible estimator (has .fit(X,y) and "
    ".predict(X)). Use ONLY numpy and sklearn (Pipelines, StandardScaler, PCA, SelectKBest, "
    "linear/SVM/tree/ensemble models, etc.). Prefer strong regularization for p>>n. "
    "Output ONLY a single ```python ...``` block defining build(); no prose."
)

_CODE_RE = re.compile(r"```(?:python)?\s*(.+?)```", re.DOTALL)

def retrieve_grounding(proposer, spec, profile="", traj=None, ctx=""):
    from retrieval import agentic_retrieve
    return agentic_retrieve(proposer, spec, profile=profile, traj=traj, ctx=ctx)


def extract_code(text: str) -> str:
    m = _CODE_RE.findall(text or "")
    return m[-1].strip() if m else (text or "").strip()


def build_from_code(code: str):
    import sklearn
    ns: dict = {"np": np, "numpy": np}
    import importlib
    for mod in ("sklearn.pipeline", "sklearn.preprocessing", "sklearn.decomposition",
                "sklearn.feature_selection", "sklearn.linear_model", "sklearn.svm",
                "sklearn.ensemble", "sklearn.tree", "sklearn.neighbors", "sklearn.base"):
        ns[mod.split(".")[-1]] = importlib.import_module(mod)
    exec(compile(code, "<codegen-module>", "exec"), ns)
    if "build" not in ns or not callable(ns["build"]):
        raise ValueError("codegen did not define a callable build()")
    return ns["build"], code


def smoke_test(build, classification=True) -> bool:
    rng = np.random.RandomState(0)
    X = rng.randn(24, 12).astype(np.float64)
    y = rng.randint(0, 2, 24) if classification else rng.randn(24)
    est = build()
    est.fit(X[:16], y[:16])
    p = np.asarray(est.predict(X[16:]))
    return p.shape[0] == 8 and np.isfinite(p.astype(float)).all()


class Registry:
    def __init__(self, seed: dict | None = None):
        self.modules: dict = {}
        for name, build in (seed or {}).items():
            self.modules[name] = {"build": build, "code": "<seed>", "oof": None, "source": "seed"}

    def register(self, name, build, code, oof, source="codegen"):
        self.modules[name] = {"build": build, "code": code, "oof": oof, "source": source}

    def names(self):
        return list(self.modules)

    def best(self):
        scored = [(n, m) for n, m in self.modules.items() if m.get("oof") is not None]
        return max(scored, key=lambda kv: kv[1]["oof"]) if scored else None


def _prompt(profile, spec, history):
    hist = "\n".join(f"  - {h}" for h in history[-8:]) or "  (none)"
    return (f"DATA REGIME:\n  {profile.get('summary', profile)}\n"
            f"Hints:\n" + "\n".join(f"  - {x}" for x in profile.get("hints", [])) + "\n\n"
            f"What to build: {spec}\n\n"
            f"Modules already tried (name -> OOF score):\n{hist}\n\n"
            "Write build() now (ONLY the ```python``` block).")


class OperationAgent:
    def __init__(self, proposer, registry: Registry, validate_fn, classification=True):
        self.proposer = proposer
        self.registry = registry
        self.validate_fn = validate_fn
        self.classification = classification
        self.history = []

    def create(self, profile, spec, name, min_score=None):
        try:
            text = self.proposer.complete(MODULE_SYSTEM, _prompt(profile, spec, self.history))
            build, code = build_from_code(extract_code(text))
        except Exception as exc:
            print(f"  [codegen] {name}: generation/exec FAILED -> {type(exc).__name__}: {exc}", flush=True)
            self.history.append(f"{name} -> codegen error")
            return None
        try:
            if not smoke_test(build, self.classification):
                print(f"  [codegen] {name}: smoke-test FAILED (bad output shape/nan)", flush=True)
                self.history.append(f"{name} -> smoke-test failed")
                return None
        except Exception as exc:
            print(f"  [codegen] {name}: smoke-test EXCEPTION -> {type(exc).__name__}: {exc}", flush=True)
            self.history.append(f"{name} -> smoke-test exception")
            return None
        try:
            oof = float(self.validate_fn(build))
        except Exception as exc:
            print(f"  [codegen] {name}: OOF-validate EXCEPTION -> {type(exc).__name__}: {exc}", flush=True)
            self.history.append(f"{name} -> validate exception")
            return None
        self.history.append(f"{name} -> OOF {oof:.4f}")
        if min_score is not None and oof < min_score:
            print(f"  [codegen] {name}: OOF {oof:.4f} < bar {min_score:.4f} -> NOT registered", flush=True)
            return oof
        self.registry.register(name, build, code, oof)
        print(f"  [codegen] {name}: OOF {oof:.4f} -> REGISTERED (registry now {len(self.registry.names())})", flush=True)
        return oof

    def create_iterative(self, profile, spec, name, max_iters=4, min_score=None):
        grounding = retrieve_grounding(self.proposer, spec)
        if grounding:
            print("  [retrieve] grounded codegen with real-method references", flush=True)
            spec = spec + f"\n\nGrounding (real published methods/repos to draw from):\n{grounding}"
        messages = [{"role": "user", "content": _prompt(profile, spec, self.history)}]
        best = None
        for it in range(max_iters):
            try:
                text = self.proposer.chat(MODULE_SYSTEM, messages)
            except Exception as exc:
                print(f"  [{name} it{it + 1}] LLM error -> {type(exc).__name__}: {exc}", flush=True)
                break
            messages.append({"role": "assistant", "content": text})
            try:
                build, code = build_from_code(extract_code(text))
                if not smoke_test(build, self.classification):
                    fb = "Your build() ran but predict() returned a bad shape or NaN. Fix and re-output build()."
                else:
                    oof = float(self.validate_fn(build))
                    if best is None or oof > best[0]:
                        best = (oof, build, code)
                    bar = f" (bar {min_score:.4f})" if min_score is not None else ""
                    ok = min_score is not None and oof >= min_score
                    print(f"  [{name} it{it + 1}] OOF={oof:.4f}{bar} {'>= bar' if ok else ''}", flush=True)
                    fb = (f"Your module scored OOF macro-F1 = {oof:.4f}{bar}. "
                          f"{'Above the bar. ' if ok else 'Below the bar. '}"
                          "Try to IMPROVE it (e.g. stronger regularization for p>>n, different model family, "
                          "PCA/SelectKBest tuning, calibrated ensemble). Output an improved build().")
            except Exception as exc:
                fb = f"Your code raised {type(exc).__name__}: {str(exc)[:200]}. Fix the code and re-output a corrected build()."
                print(f"  [{name} it{it + 1}] error -> {type(exc).__name__}: {str(exc)[:80]}", flush=True)
            messages.append({"role": "user", "content": fb})

        if best is None:
            self.history.append(f"{name} -> all {max_iters} iters failed")
            print(f"  [{name}] no valid module over {max_iters} iters", flush=True)
            return None
        oof, build, code = best
        self.history.append(f"{name} -> OOF {oof:.4f} (iter)")
        if min_score is not None and oof < min_score:
            print(f"  [{name}] best {oof:.4f} < bar {min_score:.4f} -> NOT registered", flush=True)
            return oof
        self.registry.register(name, build, code, oof, source="codegen-iter")
        print(f"  [{name}] BEST OOF {oof:.4f} over {max_iters} iters -> REGISTERED "
              f"(registry now {len(self.registry.names())})", flush=True)
        return oof
