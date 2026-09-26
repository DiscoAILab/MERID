"""Uncertainty-aware verification on development data: subject-bootstrap standard errors, rank-transfer diagnostics and the winner's-curse margin."""
from __future__ import annotations

import numpy as np


def average_ranks(values):
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def rank_transfer(matrix):
    matrix = np.asarray(matrix, float)
    out = []
    for h in range(matrix.shape[1]):
        keep = [f for f in range(matrix.shape[1]) if f != h]
        train, held = np.nanmean(matrix[:, keep], axis=1), matrix[:, h]
        ok = ~(np.isnan(train) | np.isnan(held))
        if ok.sum() < 3:
            out.append(float("nan"))
            continue
        a, b = average_ranks(train[ok]), average_ranks(held[ok])
        spread = a.std() * b.std()
        out.append(float(((a - a.mean()) * (b - b.mean())).mean() / spread)
                   if spread > 0 else float("nan"))
    return out


def partition_rows(n_rows, groups=None, k=3):
    if groups is not None and len(groups) == n_rows:
        unique = sorted({str(g) for g in groups})
        assign = {g: i % k for i, g in enumerate(unique)}
        part = np.array([assign[str(g)] for g in groups])
    else:
        part = np.arange(n_rows) % k
    return [np.where(part == i)[0] for i in range(k)]


def fold_scores(pool, validations, y, scorer):
    ids, rows = [], []
    for record in pool:
        prediction = np.asarray(record["prediction"])
        if prediction.shape != y.shape:
            continue
        ids.append(record["id"])
        rows.append([float(scorer(y[v], prediction[v])) if len(v) else float("nan")
                     for v in validations])
    return ids, np.asarray(rows, float)


def subject_bootstrap_se(y, yp, groups, scorer, kind="classification", seed=0, n_boot=1000):
    y, yp, groups = np.asarray(y), np.asarray(yp), np.asarray(groups)
    subs = np.unique(groups)
    if len(subs) < 4 or len(yp) != len(y):
        return 0.0
    idx_of = {g: np.flatnonzero(groups == g) for g in subs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(subs), len(subs))
        rows = np.concatenate([idx_of[subs[i]] for i in pick])
        if kind == "classification" and len(np.unique(y[rows])) < 2:
            continue
        try:
            out.append(float(scorer(y[rows], yp[rows])))
        except Exception:
            continue
    return float(np.std(out)) if out else 0.0


def winners_curse_margin(se, n_candidates, multiplier=1.0):
    # The margin scales with the number of compared candidates: the best of N noisy scores is
    # inflated by about se * sqrt(2 ln N).
    n = max(2, int(n_candidates))
    return float(multiplier) * float(se) * float(np.sqrt(2.0 * np.log(n)))


def assess_promotion(search):
    records = [r for r in getattr(search, "records", [])
               if r.get("valid") and r.get("score") is not None and r.get("prediction") is not None]
    references = [r for r in records if int(r.get("round", 0)) == -1]
    best = getattr(search, "best", None)
    if not references or not best or best.get("score") is None:
        return None
    reference = max(references, key=lambda r: float(r["score"]))
    if best["id"] in {r["id"] for r in references}:
        return {"applicable": False, "reason": "the deployed candidate is a reference",
                "reference_id": reference["id"], "deployed_id": best["id"],
                "n_candidates": len(records)}
    split = "dev" if getattr(search, "protocol", None) == "official_dev_selection" else "train"
    try:
        y = np.asarray(search.reference["%s_%s" % (split, search.task)])
        groups = np.asarray(search.reference["%s_subject" % split])
    except (KeyError, TypeError, AttributeError):
        return None
    se = subject_bootstrap_se(y, reference["prediction"], groups, search.scorer,
                              kind=getattr(search, "kind", "classification"),
                              seed=int(getattr(search, "seed", 0)))
    multiplier = float(getattr(search, "promotion_se", 1.0) or 0.0)
    margin = winners_curse_margin(se, len(records), multiplier)
    lead = float(best["score"]) - float(reference["score"])
    return {"applicable": True, "reference_id": reference["id"], "deployed_id": best["id"],
            "reference_score": float(reference["score"]), "deployed_score": float(best["score"]),
            "lead": lead, "reference_se": se, "n_candidates": len(records),
            "multiplier": multiplier, "margin": margin, "cleared": bool(lead >= margin)}


def assess_search(search, k=3):
    pool = [r for r in getattr(search, "records", [])
            if r.get("valid") and r.get("score") is not None and r.get("prediction") is not None]
    reference = getattr(search, "reference", None)
    if len(pool) < 3 or not reference:
        return None
    split = "dev" if getattr(search, "protocol", None) == "official_dev_selection" else "train"
    try:
        y = np.asarray(reference["%s_%s" % (split, search.task)])
        groups = reference.get("%s_subject" % split)
    except (KeyError, TypeError):
        return None

    validations = [np.asarray(v, int) for _, v in getattr(search, "folds", [])]
    synthesised = not validations
    if synthesised:
        validations = partition_rows(len(y), groups, k=k)
    if len(validations) < 3:
        return None

    ids, matrix = fold_scores(pool, validations, y, search.scorer)
    if len(ids) < 3:
        return None
    rho = rank_transfer(matrix)
    finite = [v for v in rho if not np.isnan(v)]
    if not finite:
        return None
    return {
        "criterion": "the development ranking transfers to every held-out fold (Spearman rho > 0)",
        "rank_transfer": rho,
        "worst_rho": float(min(finite)),
        "usable": bool(min(finite) > 0.0),
        "n_candidates": len(ids),
        "n_folds": len(validations),
        "folds_synthesised": bool(synthesised),
        "measured_on": split,
    }
