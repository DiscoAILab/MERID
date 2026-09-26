"""Final evaluation for E-DAIC, Mental Health and Kaggle: the deployment selected on development data, scored once on the held-out test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

import merid.run_corpus as RC
from merid.core.grounded_state_construction.adapter_agent import _build


def _complexity(config):
    if config == "codegen" or not isinstance(config, dict):
        return (3, 10 ** 6, 9)
    fam = {"svc_linear": 0, "logreg": 0, "ridge": 0, "lasso": 0, "pls": 0, "svc_rbf": 1, "svr": 1, "gb": 2, "rf": 3, "et": 3}
    return (0, int(config.get("k", 128)), fam.get(config.get("model"), 5))


def _members_1se(summary, summary_path, topk, metric, task, seed=0, n_boot=1000, honour_incumbent=True):
    from merid.run_method import metric_value
    d = os.path.dirname(os.path.abspath(summary_path))
    pf = os.path.join(d, "development_predictions.json")
    if not os.path.exists(pf):
        raise SystemExit("no development_predictions.json beside the summary; use --members top")
    q = json.load(open(pf, encoding="utf-8"))
    ids = q["identities"]
    split = "dev" if summary.get("protocol") == "official_dev_selection" and ("dev_" + task) in ids else "train"
    y = np.asarray(ids[split + "_" + task], float)
    groups = np.asarray(ids[split + "_subject"]).astype(str)
    pred_of = {c["id"]: np.asarray(c["prediction"], float) for c in q["candidates"] if c.get("prediction") is not None}
    cands = [c for c in summary.get("candidates", []) if c.get("valid", True) and c.get("score") is not None]
    sel = [c for c in cands if c.get("id") == summary.get("selected_id")]
    if not sel or sel[0]["id"] not in pred_of or len(pred_of[sel[0]["id"]]) != len(y):
        raise SystemExit("selected candidate has no stored prediction on the selection venue")
    sel = sel[0]
    yp = pred_of[sel["id"]]
    if not RC._is_reg():
        y, yp = y.astype(int), yp.astype(int)
    subs = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in subs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(subs), len(subs))
        rows = np.concatenate([idx_of[subs[i]] for i in pick])
        if not RC._is_reg() and len(np.unique(y[rows])) < 2:
            continue
        try:
            out.append(float(metric_value(metric, y[rows], yp[rows])))
        except Exception:
            continue
    se = float(np.std(out)) if out else 0.0
    floor = float(sel["score"]) - se
    tied = [c for c in cands if float(c["score"]) >= floor - 1e-12]
    tied.sort(key=lambda c: (_complexity(c["config"]), -float(c["score"])))
    members, branch = _incumbent_members(summary, sel, cands, tied, topk, honour_incumbent)
    return members, se, len(tied), split, branch


def _reference_ids(summary):
    rc = summary.get("reference_candidates") or {}
    return [r["id"] for r in rc.get("reevaluated", []) if r.get("valid") and r.get("id")]


def _incumbent_members(summary, sel, cands, tied, topk, honour_incumbent=True):
    refs = _reference_ids(summary)
    topk = max(1, topk)
    if not honour_incumbent or not refs or sel["id"] not in refs:
        return tied[:topk], ("promoted -> 1-SE tied set of the whole pool" if refs else
                             "no reference candidates -> 1-SE tied set of the whole pool")
    ref_set = set(refs)
    if len(refs) <= topk:
        by_id = {c["id"]: c for c in cands}
        members = [by_id[i] for i in refs if i in by_id]
        return members, "incumbent retained -> reference members as listed"
    members = [c for c in tied if c["id"] in ref_set][:1]
    return members, "incumbent retained -> reference rule inside the reference grid (simplest inside 1 SE, one model)"


def _winners(summary, topk):
    cands = [c for c in summary.get("candidates", []) if c.get("valid", True) and c.get("score") is not None]
    sel = [c for c in cands if c.get("id") == summary.get("selected_id")]
    if not sel:
        raise SystemExit("selected_id %r is not among the valid candidates" % summary.get("selected_id"))
    rest = sorted((c for c in cands if c is not sel[0]), key=lambda c: float(c["score"]), reverse=True)
    return (sel + rest)[:max(1, topk)]


def _codegen_builder(summary_path, winners):
    if not any(c["config"] == "codegen" for c in winners):
        return None, None
    path = os.path.join(os.path.dirname(os.path.abspath(summary_path)), "codegen_model.py")
    if not os.path.isfile(path):
        raise SystemExit("a member is the codegen candidate but %s is missing, so it cannot be rebuilt" % path)
    from merid.core.coupled_pipeline_exploration.operation import build_from_code
    with open(path, encoding="utf-8") as fh:
        code = fh.read()
    build, _code = build_from_code(code)
    return build, hashlib.sha256(code.encode()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, help="development_summary.json of the development run")
    ap.add_argument("--adapter", required=True, help="the frozen adapter .py that run used")
    ap.add_argument("--ctx", required=True, help="corpus name as run_corpus knew it (e.g. E-DAIC)")
    ap.add_argument("--task", default=None, help="defaults to the summary's task")
    ap.add_argument("--regression_tasks", default="phq,score,severity,total")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--members", choices=["1se", "1se_pool", "top"], default="1se",
                    help="1se = the reference members when the incumbent was retained, else the 1-SE tied "
                         "set (simplest first, at most topk); 1se_pool = the 1-SE tied set of the whole pool "
                         "regardless of the incumbent (ablation); top = the topk highest development scores")
    ap.add_argument("--allow_adapter_mismatch", action="store_true",
                    help="evaluate with an adapter whose sha256 differs from the one the development run "
                         "recorded. Needed when that run's adapter was the development-scope repair (it omits "
                         "test loading by design) and the full-scope original must be used to reach the test "
                         "block; both hashes are written into the result.")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)

    summary = json.load(open(a.summary, encoding="utf-8"))
    code = open(a.adapter, encoding="utf-8").read()
    sha = hashlib.sha256(code.encode()).hexdigest()
    want = summary.get("adapter_source_sha256")
    if want and want != sha:
        msg = ("adapter mismatch: the summary was produced with sha256 %s..., this file is %s..."
               % (want[:12], sha[:12]))
        if not a.allow_adapter_mismatch:
            raise SystemExit(msg + " (pass --allow_adapter_mismatch if this is the full-scope original of a "
                                   "development-scope repair)")
        print("   WARNING " + msg + " - proceeding because --allow_adapter_mismatch was given", flush=True)
    RC.CTX = a.ctx
    RC.TASK = a.task or summary["task"]
    RC.REG_TASKS = tuple(t.strip() for t in a.regression_tasks.split(",") if t.strip())
    RC.SEL_NAME = (summary.get("primary_metric") or RC.SEL_NAME)
    if a.members in ("1se", "1se_pool"):
        winners, se, n_tied, venue, branch = _members_1se(summary, a.summary, a.topk, RC.SEL_NAME, RC.TASK,
                                                          honour_incumbent=(a.members == "1se"))
        rule = "%s on %s (SE %.4f, %d tied, %d used)" % (branch, venue, se, n_tied, len(winners))
    else:
        winners = _winners(summary, a.topk)
        rule = "top-%d by development score" % len(winners)
    print("== %s / %s: published-test evaluation of the development winner (%s) ==" % (RC.CTX, RC.TASK, rule))
    print("   summary: %s  (%s, selected %s = %.4f on %s)" % (
        a.summary, summary.get("protocol"), RC.SEL_NAME, float(summary.get("primary_score") or 0), summary.get("query_split")))
    for i, c in enumerate(winners):
        print("   #%d dev %.4f  recipe=%s  config=%s" % (i + 1, float(c["score"]), c["recipe"], c["config"]))
    if a.dry_run:
        return

    t0 = time.time()
    codegen_build, codegen_sha = _codegen_builder(a.summary, winners)
    load = _build(code)
    preds_t, preds_d, yt, test_ids = [], [], None, None
    for c in winners:
        data = load(c["recipe"])
        if data.get("test_ids") is not None:
            ids = [str(i) for i in data["test_ids"]]
            if test_ids is not None and ids != test_ids:
                raise SystemExit("members disagree on test_ids order; predictions could not be joined")
            test_ids = ids
        for k in ("train_X", "test_X", "test_" + RC._ykey()):
            if k not in data:
                raise SystemExit("the adapter returned no %r: this corpus publishes no test block, or the "
                                 "adapter was written for development scope only" % k)
        X, y = np.asarray(data["train_X"], float), RC._y(data, "train")
        Xt = np.asarray(data["test_X"], float)
        yt = RC._y(data, "test")
        preds_t.append(RC._fit_predict(c["config"], codegen_build, X, y, Xt))
        if RC._has_dev(data):
            preds_d.append((RC._fit_predict(c["config"], codegen_build, X, y, np.asarray(data["dev_X"], float)),
                            RC._y(data, "dev")))
    P = np.vstack(preds_t)
    if RC._is_reg():
        yp = P.mean(axis=0)
    else:
        yp = np.apply_along_axis(lambda col: np.bincount(col.astype(int)).argmax(), 0, P)
    from merid.run_method import metric_value as _mv
    RC._report(yt, yp, len(yt), "published test, fit on train only (development winner; %s)" % rule,
               [{"recipe": c["recipe"], "config": c["config"]} for c in winners], None)
    gap = None
    if preds_d:
        Pd = np.vstack([p for p, _ in preds_d])
        ydp = Pd.mean(axis=0) if RC._is_reg() else np.apply_along_axis(lambda col: np.bincount(col.astype(int)).argmax(), 0, Pd)
        dev_score = _mv(RC.SEL_NAME, preds_d[0][1], ydp)
        test_score = _mv(RC.SEL_NAME, yt, yp)
        gap = float(test_score - dev_score)
        print("  selection (dev) %s = %.4f | test = %.4f | gap %+.4f" % (RC.SEL_NAME, dev_score, test_score, gap), flush=True)
    print("   done in %.0fs" % (time.time() - t0))

    from merid.run_method import METRICS, REGRESSION_METRICS
    names = REGRESSION_METRICS if RC._is_reg() else METRICS
    out = os.path.join(os.path.dirname(os.path.abspath(a.summary)),
                       "final_test_metrics_%s%d.json" % ("1se" if a.members == "1se" else "top", len(winners)))
    json.dump({"ctx": RC.CTX, "task": RC.TASK, "selection_metric": RC.SEL_NAME, "n_test": int(len(yt)), "member_rule": rule,
               "topk": len(winners), "winners": [{"recipe": c["recipe"], "config": c["config"], "dev_score": c["score"]} for c in winners],
               "metrics": {n: float(_mv(n, yt, yp)) for n in names}, "selection_to_test_gap": gap,
               "test_ids": test_ids,
               "y_true": [(float(v) if RC._is_reg() else int(v)) for v in np.asarray(yt).ravel()],
               "prediction": [(float(v) if RC._is_reg() else int(v)) for v in np.asarray(yp).ravel()],
               "member_prediction": [[(float(v) if RC._is_reg() else int(v)) for v in np.asarray(row).ravel()] for row in P],
               "adapter_sha256": sha, "development_adapter_sha256": want, "adapter_matches_development": (want == sha),
               "codegen_sha256": codegen_sha,
               "summary_sha256": hashlib.sha256(open(a.summary, "rb").read()).hexdigest(),
               "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(out, "w", encoding="utf-8"), indent=2)
    print("   metrics -> %s" % out)


if __name__ == "__main__":
    main()
