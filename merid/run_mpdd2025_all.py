"""MPDD-2025 helpers: official metric, grouped folds and classifier factories."""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter_agent import AdapterAgent
from proposer import ClaudeProposer
from run_adapter import _svc, clf_candidate, folds_of

CONFIGS = [
    ("MPDD-Young", "MPDD-Young", "MM2025_Track2_Young.json",
     [("binary", "label_bin"), ("ternary", "label_tri")]),
    ("MPDD-Elderly", "MPDD-Elderly", "MM2025_Track1_Elderly.json",
     [("binary", "label_bin"), ("ternary", "label_tri"), ("quinary", "label_pen")]),
]
WINDOWS = ["1s", "5s"]


import json
import re

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLF_SPACE = ('{"model": one of ["svc_linear","svc_rbf","logreg","gb","rf","et"], '
             '"k": one of [32,64,128,256] (SelectKBest features), '
             '"C": for svc/logreg e.g. 0.05..1.0, "lr": for gb e.g. 0.03..0.1, "depth": for gb 1..3}')
_J = re.compile(r"\{[^{}]*\}")


def make_clf(cfg, p):
    from sklearn.feature_selection import VarianceThreshold
    k = int(cfg.get("k", 128)); m = cfg.get("model", "svc_linear")
    steps = [("v", VarianceThreshold(0.0)), ("s", StandardScaler())]
    if 0 < k < p:
        steps.append(("sel", SelectKBest(f_classif, k=min(k, p))))
    if m == "svc_rbf":
        clf = SVC(C=float(cfg.get("C", 1.0)), kernel="rbf", class_weight="balanced")
    elif m == "logreg":
        clf = LogisticRegression(C=float(cfg.get("C", 0.1)), class_weight="balanced", max_iter=3000)
    elif m == "gb":
        clf = GradientBoostingClassifier(learning_rate=float(cfg.get("lr", 0.05)),
                                         max_depth=int(cfg.get("depth", 1)), n_estimators=120, random_state=0)
    elif m == "rf":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0)
    elif m == "et":
        clf = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=0)
    else:
        clf = SVC(C=float(cfg.get("C", 0.1)), kernel="linear", class_weight="balanced")
    return Pipeline(steps + [("clf", clf)])


def make_reg(cfg, p):
    from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                                  RandomForestRegressor)
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    k = int(cfg.get("k", 128)); m = cfg.get("model", "ridge")
    steps = [("v", VarianceThreshold(0.0)), ("s", StandardScaler())]
    if 0 < k < p:
        steps.append(("sel", SelectKBest(f_regression, k=min(k, p))))
    if m == "svr_rbf":
        r = SVR(C=float(cfg.get("C", 1.0)), kernel="rbf", epsilon=float(cfg.get("eps", 1.0)))
    elif m == "svr_linear":
        r = SVR(C=float(cfg.get("C", 0.3)), kernel="linear", epsilon=float(cfg.get("eps", 1.0)))
    elif m == "gbr":
        r = GradientBoostingRegressor(learning_rate=float(cfg.get("lr", 0.05)),
                                      max_depth=int(cfg.get("depth", 2)), n_estimators=150,
                                      random_state=0)
    elif m == "rfr":
        r = RandomForestRegressor(n_estimators=300, random_state=0)
    elif m == "etr":
        r = ExtraTreesRegressor(n_estimators=300, random_state=0)
    else:
        r = Ridge(alpha=float(cfg.get("alpha", 10.0)))
    return Pipeline(steps + [("reg", r)])


def grouped_folds(y, groups, k=5, seed=0):
    if groups is None:
        return folds_of(y)
    groups = np.asarray(groups)
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        ns = max(2, min(k, len(np.unique(groups)) // 2))
        return list(StratifiedGroupKFold(n_splits=ns, shuffle=True, random_state=seed).split(np.zeros((len(y), 1)), y, groups))
    except Exception:
        from sklearn.model_selection import GroupKFold
        ns = max(2, min(k, len(np.unique(groups))))
        return list(GroupKFold(n_splits=ns).split(np.zeros((len(y), 1)), y, groups))


def oof_clf(X, Xt, y, cfg, folds):
    from scipy import stats
    p = X.shape[1]; oof = np.full(len(y), -1); tp = []
    for tr, va in folds:
        if len(set(y[tr].tolist())) < 2:
            continue
        try:
            m = make_clf(cfg, p); m.fit(X[tr], y[tr]); oof[va] = m.predict(X[va]); tp.append(m.predict(Xt))
        except Exception:
            continue
    if not tp:
        return oof, np.full(Xt.shape[0], int(np.bincount(y).argmax())), 0.0
    test = np.asarray(stats.mode(np.array(tp), axis=0, keepdims=False).mode).ravel()
    mk = oof >= 0
    return oof, test, (f1_score(y[mk], oof[mk], average="macro", zero_division=0) if mk.any() else 0.0)


def _parse_cfg(text):
    for c in reversed(_J.findall(text or "")):
        try:
            o = json.loads(c)
            if isinstance(o, dict) and "model" in o:
                return o
        except Exception:
            continue
    return {"model": "svc_linear", "k": 128, "C": 0.1}


def agent_classify(X, Xt, y, proposer, prof, task, groups=None, budget=4, topk=3, traj=None, ctx=""):
    from scipy import stats
    y = np.asarray(y, int)
    if len(set(y.tolist())) < 2:
        return np.full(Xt.shape[0], int(y[0]))
    folds = grouped_folds(y, groups)
    sysmsg = ("You choose a scikit-learn classifier CONFIG for a SMALL-SAMPLE p>>n multimodal "
              "depression task. Reason from the data regime, then output ONLY a JSON config: " + CLF_SPACE)
    hist, cands = [], []
    for i in range(budget):
        reasoning = ""
        try:
            txt = proposer.complete(sysmsg, f"DATA REGIME: {prof}\nTask: {task}\nTried: {hist or '(none)'}\n"
                                            f"Propose the next classifier JSON config (final line).")
            reasoning = txt; cfg = _parse_cfg(txt)
        except Exception:
            cfg = {"model": "svc_linear", "k": 128, "C": 0.1}
        oof, test, f = oof_clf(X, Xt, y, cfg, folds)
        cov = float((np.asarray(oof) >= 0).mean())
        warn = "  !! LOW COVERAGE (score is on a subset)" if cov < 0.9 else ""
        hist.append(f"{cfg} -> OOF F1 {f:.3f} (cov {cov:.2f})")
        cands.append((f, test, json.dumps(cfg, sort_keys=True)))
        print(f"    [clf {task} {i + 1}/{budget}] {cfg} -> OOF F1 {f:.4f} cov={cov:.2f}{warn}", flush=True)
        if traj is not None:
            traj.log("classifier_propose", reasoning=reasoning, dataset=ctx, task=task,
                     choice=cfg, oof_macroF1=round(float(f), 4), oof_coverage=round(cov, 4))
    cands.sort(key=lambda z: z[0], reverse=True)
    seen, uniq = set(), []
    for f, test, key in cands:
        if key in seen:
            continue
        seen.add(key); uniq.append(test)
    if len(uniq) < len(cands):
        print(f"    [clf {task}] deduped {len(cands)} proposals -> {len(uniq)} distinct configs", flush=True)
    return np.asarray(stats.mode(np.array(uniq[:topk]), axis=0, keepdims=False).mode).ravel()


def official(yt, yp):
    yt, yp = np.asarray(yt, int), np.asarray(yp, int)
    acc = accuracy_score(yt, yp); wacc = balanced_accuracy_score(yt, yp)
    mf1 = f1_score(yt, yp, average="macro", zero_division=0); wf1 = f1_score(yt, yp, average="weighted", zero_division=0)
    return acc, wacc, mf1, wf1, ((acc + wacc) / 2 + (mf1 + wf1) / 2) / 2


def spec_for(track, window, mm, tasks):
    has_q = any(t == "quinary" for t, _ in tasks)
    quin = (" Also return train_quinary (5-class) from pen_category and test_quinary from label_pen.") if has_q else ""
    qfield = ", pen_category" if has_q else ""
    return (f"MPDD-2025 {track} {window} window, per-SEGMENT CLASSIFICATION (no PHQ). "
            f"TRAIN: per-file labels are in <train_root>/labels/Training_Validation_files.json = list of dicts with "
            f"{{id, audio_feature_path, video_feature_path, bin_category, tri_category{qfield}}}. Audio features live under "
            f"<train_root>/{window}/Audio/<AUDIO_EXT>/ and video under <train_root>/{window}/Visual/<VIDEO_EXT>/ — the "
            f"AVAILABLE extractors + their shapes are in the inspection above. Do NOT choose the recipe: implement the "
            f"`recipe` dict FAITHFULLY (AUDIO_EXT=recipe['audio']='wav2vec'; VIDEO_EXT=recipe['video']; aggregation="
            f"recipe['agg']; append per-subject individualEmbedding iff recipe['ind_emb'] — the individualEmbedding dir/"
            f"file is shown in the inspection, keyed by subject id). If recipe['meta']: also append the 8-dim structured "
            f"metadata (big5_traits + family_factors + disease) read from personalized_train.json (train, under "
            f"<train_root> or <train_root>/labels) and personalized_test.json (test, under <test_root>), each a dict keyed "
            f"by subject id. An outer loop OOF-selects the best recipe. Match each "
            f"label entry to its feature file by "
            f"audio_feature_path/video_feature_path (or same basename). Labels: bin_category->binary, tri_category->ternary"
            f"{', pen_category->quinary' if has_q else ''}. "
            f"TEST: for each entry in <test_root>/{mm} (list of {{test_id, label_bin, label_tri"
            f"{', label_pen' if has_q else ''}}}), load the SAME chosen extractors from "
            f"<test_root>/{track}/{window}/Audio/<AUDIO_EXT>/<test_id>.npy + Visual/<VIDEO_EXT>/<test_id>.npy. "
            f"Return train_X/test_X (same p, your chosen recipe), train_binary/ternary + test_binary/ternary, test_ids, "
            f"AND train_subject = the subject id of each train segment (the part of the filename before the FIRST underscore, "
            f"e.g. '4173' from '4173_A_1') so cross-validation can group by subject and avoid leakage.{quin} No PHQ.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="<MPDD2025_ROOT>")
    ap.add_argument("--test_root", required=True, help="<MPDD2025_TEST_ROOT>")
    ap.add_argument("--llm_model", default="claude-opus-4-8")
    ap.add_argument("--proposer", default="anthropic", choices=["anthropic"])
    ap.add_argument("--llm_api_key", default=None)
    ap.add_argument("--llm_base_url", default=None)
    a = ap.parse_args()

    proposer = ClaudeProposer(model=a.llm_model, provider=a.proposer, base_url=a.llm_base_url,
                              api_key=a.llm_api_key)
    agent = AdapterAgent(proposer)
    from adapter_agent import ADAPTER_SYSTEM
    from trajectory import Trajectory
    traj = Trajectory(model=a.llm_model, config={"task": "MPDD-2025 classification"},
                      stream_path="mpdd2025_trajectory.jsonl")
    traj.add_system_prompt("adapter_codegen", ADAPTER_SYSTEM)
    traj.add_system_prompt("classifier_propose",
                           "Choose a scikit-learn classifier CONFIG for SMALL-SAMPLE p>>n; reason then JSON.")
    proposer.traj = traj

    rows = []
    for track, test_sub, mm, tasks in CONFIGS:
        train_root = os.path.join(a.data_root, track, "Training")
        for window in WINDOWS:
            ctx = f"MPDD2025/{track}/{window}"
            print(f"\n########## {track} {window} — agent writes adapter ##########", flush=True)
            spec = spec_for(track, window, mm, tasks)
            load, code, data, adapter_reasoning = agent.create_adapter(
                {"train_root": train_root, "test_root": a.test_root}, spec, max_iters=5)
            traj.log("adapter_codegen", reasoning=adapter_reasoning, dataset=ctx,
                     ok=load is not None, code=code)
            if load is None:
                print(f"  [{track} {window}] adapter FAILED"); continue
            X, Xt = np.asarray(data["train_X"], float), np.asarray(data["test_X"], float)
            print(f"  adapter OK: train {X.shape} test {Xt.shape}", flush=True)
            import collections
            dists = {t: dict(sorted(collections.Counter(np.asarray(data[f"train_{t}"], int).tolist()).items()))
                     for t, _ in tasks if f"train_{t}" in data}
            groups = data.get("train_subject")
            ng = len(np.unique(groups)) if groups is not None else 0
            prof = (f"{track} {window}: n_train={X.shape[0]} segments from {ng} subjects, p={X.shape[1]} "
                    f"(p>>n={X.shape[1] > X.shape[0]}); class dists {dists}")
            print(f"  REGIME: {prof}  (grouped CV by subject: {'yes' if groups is not None else 'NO - leakage risk'})", flush=True)
            traj.log("regime_profile", reasoning="", dataset=ctx, profile=prof, n_subjects=int(ng))
            for task, _lk in tasks:
                trk, tek = f"train_{task}", f"test_{task}"
                if trk not in data or tek not in data:
                    print(f"  {task}: adapter missing {trk}/{tek}, skip"); continue
                yp = agent_classify(X, Xt, np.asarray(data[trk], int), proposer, prof, task,
                                    groups=groups, traj=traj, ctx=ctx)
                acc, wacc, mf1, wf1, off = official(data[tek], yp)
                rows.append((track, window, task, len(yp), acc, wacc, mf1, wf1, off))
                traj.log("result", reasoning="", dataset=ctx, task=task,
                         official=round(off, 4), acc=round(acc, 4), macroF1=round(mf1, 4))
                print(f"  {task:<9} Acc={acc:.4f} WAcc={wacc:.4f} MF1={mf1:.4f} WF1={wf1:.4f} Official={off:.4f}", flush=True)

    traj.close("mpdd2025_trajectory.json")
    ms = traj.info["model_stats"]
    print(f"\n[trajectory] {ms['api_calls']} LLM calls, {ms['output_tokens']} out-tokens, "
          f"${ms['cost']:.3f} -> mpdd2025_trajectory.json")
    print("\n==================== MERID — MPDD-2025 ====================")
    print(f"{'Track':<13}{'Win':<4}{'Task':<9}{'N':>5}{'Acc':>8}{'WAcc':>8}{'MacroF1':>9}{'WF1':>8}{'Official':>9}")
    for tr, w, tk, n, acc, wacc, mf1, wf1, off in rows:
        print(f"{tr:<13}{w:<4}{tk:<9}{n:>5}{acc:>8.4f}{wacc:>8.4f}{mf1:>9.4f}{wf1:>8.4f}{off:>9.4f}")


if __name__ == "__main__":
    main()
