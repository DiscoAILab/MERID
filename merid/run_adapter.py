"""Generic classical method on an agent-written adapter, plus the shared classifier and fold helpers."""
import argparse
import os
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter_agent import AdapterAgent
from proposer import ClaudeProposer


def folds_of(y, k=10, seed=0):
    y = np.asarray(y, int)
    counts = np.bincount(y)
    mn = int(counts[counts > 0].min())
    ns = max(2, min(k, mn))
    try:
        return list(StratifiedKFold(n_splits=ns, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y))
    except ValueError:
        from sklearn.model_selection import KFold
        return list(KFold(n_splits=max(2, min(k, len(y) // 2)), shuffle=True, random_state=seed).split(np.zeros((len(y), 1))))


def _pls(k, nc, p):
    steps = [("v", VarianceThreshold(0.0)), ("s", StandardScaler())]
    if 0 < k < p:
        steps.append(("k", SelectKBest(f_regression, k=min(k, p))))
    steps.append(("m", PLSRegression(n_components=nc)))
    return Pipeline(steps)


def _svc(C, k, p):
    steps = [("v", VarianceThreshold(0.0)), ("s", StandardScaler())]
    if 0 < k < p:
        steps.append(("k", SelectKBest(f_classif, k=min(k, p))))
    steps.append(("m", SVC(C=C, kernel="linear", class_weight="balanced", probability=False)))
    return Pipeline(steps)


def _calib(oof, phq, shrink):
    ids = np.arange(len(phq))
    pm, ps, tm, ts = oof.mean(), oof.std() + 1e-9, phq.mean(), phq.std()
    direction = 1.0 if np.corrcoef(oof, phq)[0, 1] >= 0 else -1.0
    scale = direction * shrink * ts / ps
    return lambda x: np.clip(scale * x + (tm - scale * pm), 0, 27)


def ccc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    d = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    return float(2 * cov / d) if d > 1e-10 else 0.0


def phq_candidate(X, Xt, phq, folds, k, nc, shrink):
    p = X.shape[1]; oof = np.zeros(len(phq)); tp = np.zeros(Xt.shape[0]); nf = 0
    for tr, va in folds:
        m = _pls(k, nc, p); m.fit(X[tr], phq[tr])
        oof[va] = np.clip(m.predict(X[va]).ravel(), 0, 27)
        tp += np.clip(m.predict(Xt).ravel(), 0, 27); nf += 1
    tp /= nf
    cal = _calib(oof, phq, shrink)
    return cal(oof), cal(tp), ccc(phq, cal(oof))


def clf_candidate(X, Xt, y, folds, C, k):
    p = X.shape[1]; oof = np.full(len(y), -1); tp = []
    for tr, va in folds:
        if len(set(y[tr].tolist())) < 2:
            continue
        try:
            m = _svc(C, k, p); m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va]); tp.append(m.predict(Xt))
        except Exception:
            continue
    if not tp:
        maj = int(np.bincount(y).argmax())
        return oof, np.full(Xt.shape[0], maj), 0.0
    from scipy import stats
    test = stats.mode(np.array(tp), axis=0, keepdims=False).mode
    m2 = oof >= 0
    f = f1_score(y[m2], oof[m2], average="macro", zero_division=0) if m2.any() else 0.0
    return oof, np.asarray(test).ravel(), f


def robust_method(data, topk=3):
    X = np.asarray(data["train_X"], float); Xt = np.asarray(data["test_X"], float)
    yb = np.asarray(data["train_binary"], int); yt = np.asarray(data["train_ternary"], int)
    phq = np.asarray(data["train_phq"], float)
    fb = folds_of(yb); ft = folds_of(yt)

    phqs = []
    for k in (64, 128):
        for nc in (2, 3):
            for sh in (1.2, 1.35):
                oof, test, c = phq_candidate(X, Xt, phq, fb, k, nc, sh)
                phqs.append((c, oof, test))
    phqs.sort(key=lambda z: z[0], reverse=True)
    top = phqs[:topk]
    oof_phq = np.mean([o for _, o, _ in top], 0); test_phq = np.mean([t for _, _, t in top], 0)
    grid = np.percentile(oof_phq, np.arange(20, 81, 5))
    thr = max(grid, key=lambda t: f1_score(yb, (oof_phq >= t).astype(int), average="macro", zero_division=0))
    test_bin = (test_phq >= thr).astype(int)

    ters = []
    for C in (0.1, 1.0):
        for k in (64, 128):
            oof, test, f = clf_candidate(X, Xt, yt, ft, C, k)
            ters.append((f, test))
    ters.sort(key=lambda z: z[0], reverse=True)
    from scipy import stats
    test_ter = stats.mode(np.array([t for _, t in ters[:topk]]), axis=0, keepdims=False).mode
    return {"binary": test_bin, "ternary": np.asarray(test_ter).ravel(), "phq": test_phq}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--test_root", required=True)
    ap.add_argument("--spec", required=True, help="short description of the dataset for the agent")
    ap.add_argument("--llm_model", default="claude-opus-4-8")
    ap.add_argument("--proposer", default="anthropic", choices=["anthropic"])
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--llm_api_key", default=None)
    a = ap.parse_args()

    prov = a.proposer
    proposer = ClaudeProposer(model=a.llm_model, provider=prov, base_url=a.llm_base_url,
                              api_key=a.llm_api_key)
    agent = AdapterAgent(proposer)
    load, code, data, _reasoning = agent.create_adapter({"train_root": a.train_root, "test_root": a.test_root}, a.spec)
    if load is None:
        print("AGENT FAILED to write a working adapter."); return

    print("\n== AGENT-WRITTEN ADAPTER (verify correctness, not just shapes) ==", flush=True)
    print(code, flush=True)
    print(f"\n  sizes: train_X {np.asarray(data['train_X']).shape} | test_X {np.asarray(data['test_X']).shape}"
          f" | train_binary dist {np.bincount(np.asarray(data['train_binary'], int)).tolist()}"
          f" | test_binary present {'test_binary' in data}", flush=True)

    print("\n== FIXED generic method on the agent-written adapter ==", flush=True)
    preds = robust_method(data)
    if "test_binary" not in data:
        print("adapter returned no test labels -> predictions only, cannot score."); return
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    def official(yt, yp):
        yt, yp = np.asarray(yt, int), np.asarray(yp, int)
        acc = accuracy_score(yt, yp); wacc = balanced_accuracy_score(yt, yp)
        mf1 = f1_score(yt, yp, average="macro", zero_division=0); wf1 = f1_score(yt, yp, average="weighted", zero_division=0)
        return acc, wacc, mf1, wf1, ((acc + wacc) / 2 + (mf1 + wf1) / 2) / 2

    print(f"  scored {len(data['test_ids'])} test ids  (MPDD-2025 official per-task metric)")
    print(f"  {'Task':<9}{'Acc':>8}{'WAcc':>8}{'MacroF1':>8}{'WF1':>8}{'Official':>9}")
    for task, key in [("binary", "test_binary"), ("ternary", "test_ternary")]:
        acc, wacc, mf1, wf1, off = official(data[key], preds[task])
        print(f"  {task:<9}{acc:>8.4f}{wacc:>8.4f}{mf1:>8.4f}{wf1:>8.4f}{off:>9.4f}")
    if "test_phq" in data and np.any(np.asarray(data["test_phq"], float) != 0):
        print(f"  PHQ  CCC = {ccc(data['test_phq'], preds['phq']):.4f}")
    else:
        print("  (no test PHQ in this dataset -> classification only)")


if __name__ == "__main__":
    main()
