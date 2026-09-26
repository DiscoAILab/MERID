"""Data profile (size, dimensionality, class balance, PHQ spread) shown to the proposer."""
from __future__ import annotations

import collections
import warnings

import numpy as np

MODALITIES = {
    "audio":       {"kind": "sequence", "shape": "[PAIR,T,D]", "note": "opensmile/wav2vec/mfcc"},
    "video":       {"kind": "sequence", "shape": "[PAIR,T,D]", "note": "openface/densenet/resnet"},
    "gait":        {"kind": "channel_time", "shape": "[T,9]", "note": "IMU 9-dim"},
    "personality": {"kind": "vector", "shape": "[1024]", "note": "RoBERTa text embedding"},
}


def _imbalance(dist: dict) -> float:
    c = list(dist.values())
    return round(max(c) / max(min(c), 1), 2)


def profile(harness, ref=("opensmile", "openface", "stats_q_pz")) -> dict:
    from sklearn.feature_selection import f_regression

    bank = harness._bank(*ref)
    X = np.asarray(bank["train_x"], dtype=np.float64)
    ids = list(bank["train_ids"])
    n, p = X.shape
    yb = np.array([harness.labels["binary"][i] for i in ids])
    yt = np.array([harness.labels["ternary"][i] for i in ids])
    phq = np.array([harness.phq[i] for i in ids], dtype=float)

    bdist = dict(sorted(collections.Counter(yb.tolist()).items()))
    tdist = dict(sorted(collections.Counter(yt.tolist()).items()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, _ = f_regression(X, phq)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    top = int(min(20, len(F)))

    development = getattr(harness, 'data_scope', 'all') == 'development'
    if development and 'test_ids' in bank:
        raise ValueError('development profiler received test identities')
    prof = {
        "data_scope": "development" if development else "all",
        "n_train": n, "n_test": None if development else len(bank["test_ids"]), "p_features": int(p),
        "p_over_n": round(p / max(n, 1), 1), "small_data": n < 200, "p_gg_n": p > n,
        "modalities": MODALITIES,
        "binary_dist": bdist, "binary_imbalance": _imbalance(bdist),
        "ternary_dist": tdist, "ternary_imbalance": _imbalance(tdist),
        "phq": {"mean": round(float(phq.mean()), 2), "std": round(float(phq.std()), 2),
                "min": float(phq.min()), "max": float(phq.max())},
        "univariate_signal_topF": round(float(np.sort(F)[-top:].mean()), 2),
        "ref_bank": "/".join(ref),
    }
    if development:
        try:
            from .mpdd2026_metrics import METRIC_CONTRACT
        except ImportError:
            from mpdd2026_metrics import METRIC_CONTRACT
        prof["metric_contract"] = METRIC_CONTRACT
        prof["selection_metric"] = (
            "(binary Macro-F1 + ternary Macro-F1 + raw PHQ CCC on 0..27 values) / 3")
    prof["hints"] = _hints(prof)
    prof["summary"] = _summary(prof)
    return prof


def _hints(p: dict) -> list:
    h = []
    if p["small_data"]:
        h.append(f"n={p['n_train']} is SMALL -> deep nets overfit; prefer CLASSICAL (PLS/SVM); "
                 f"consider augmentation (mixup/SMOTE) or cross-dataset pooling (missing-modality MoE, e.g. Flex-MoE).")
    if p["p_gg_n"]:
        h.append(f"p({p['p_features']})>>n({p['n_train']}) -> heavy dim-reduction (PLS n_comp small / SelectKBest low k) + shrink.")
    tern_small = min(p["ternary_dist"].values()) if p["ternary_dist"] else 0
    if p["ternary_imbalance"] >= 3 or tern_small <= 12:
        h.append(f"ternary imbalance {p['ternary_imbalance']} (rarest class n={tern_small}) -> "
                 f"class-weight + HIERARCHICAL severity head (positives-only); watch severe-class collapse.")
    if p["phq"]["std"] > 0:
        h.append("PHQ regression is the lagging task historically -> calibrate+shrink the OOF PHQ; "
                 "binary can be derived by thresholding PHQ.")
    return h


def _summary(p: dict) -> str:
    query = ('development grouped OOF; no test split' if p.get('data_scope') == 'development'
             else 'test %s' % p['n_test'])
    metric = (f"; metric={p['selection_metric']} [{p['metric_contract']}]"
              if p.get('data_scope') == 'development' else '')
    return (f"n_train={p['n_train']} ({query}), p={p['p_features']} (p/n={p['p_over_n']}, "
            f"p>>n={p['p_gg_n']}); modalities={list(p['modalities'])}; "
            f"binary={p['binary_dist']} (imb {p['binary_imbalance']}), "
            f"ternary={p['ternary_dist']} (imb {p['ternary_imbalance']}); "
            f"PHQ mean {p['phq']['mean']}+/-{p['phq']['std']} [{p['phq']['min']},{p['phq']['max']}]; "
            f"signal(topF)={p['univariate_signal_topF']}{metric}.")
