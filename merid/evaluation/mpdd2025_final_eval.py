"""Final evaluation for MPDD-2025: the deployment selected on development data, scored once on the official test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from merid.core.grounded_state_construction.adapter_agent import _build
from merid.benchmarks.mpdd2025.decoding import GroupedDecodingClassifier
from merid.benchmarks.mpdd2025.model_bridge import build_model
from merid.run_method import _subject_of, metric_value
from merid.benchmarks.mpdd2025.run_all import official


def _complexity(config):
    strategy = "plain"
    base = config
    if isinstance(config, dict) and "_decoder_strategy" in config:
        strategy, base = config["_decoder_strategy"], config.get("_base_config") or {}
    if isinstance(base, dict) and base.get("model") == "ordinal_frankhall":
        return (2, 10 ** 6, 9, 0)
    fam = {"svc_linear": 0, "logreg": 0, "svc_rbf": 1, "gb": 2, "rf": 3, "et": 3}
    order = {"plain": 0, "subject_consensus": 1, "metric_bias": 2, "metric_bias_consensus": 3}
    return (0, int(base.get("k", 128)) if isinstance(base, dict) else 10 ** 6,
            fam.get(base.get("model") if isinstance(base, dict) else None, 5), order.get(strategy, 4))


def _members_1se(summary, topk, metric, task, seed=0, n_boot=1000, honour_incumbent=True):
    d = os.path.dirname(os.path.abspath(summary["_path"]))
    pf = os.path.join(d, "development_predictions.json")
    if not os.path.exists(pf):
        raise SystemExit("no development_predictions.json beside the summary; use --members top")
    q = json.load(open(pf, encoding="utf-8"))
    ids = q["identities"]
    y = np.asarray(ids["train_" + task], int)
    groups = np.asarray(ids["train_subject"]).astype(str)
    pred_of = {c["id"]: np.asarray(c["prediction"], int) for c in q["candidates"] if c.get("prediction") is not None}
    cands = [c for c in summary.get("candidates", []) if c.get("valid", True) and c.get("score") is not None]
    sel = [c for c in cands if c.get("id") == summary.get("selected_id")]
    if not sel or sel[0]["id"] not in pred_of:
        raise SystemExit("selected candidate has no stored out-of-fold prediction")
    sel = sel[0]
    yp = pred_of[sel["id"]]
    subs = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in subs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(subs), len(subs))
        rows = np.concatenate([idx_of[subs[i]] for i in pick])
        if len(np.unique(y[rows])) < 2:
            continue
        out.append(float(metric_value(metric, y[rows], yp[rows])))
    se = float(np.std(out)) if out else 0.0
    floor = float(sel["score"]) - se
    tied = [c for c in cands if float(c["score"]) >= floor - 1e-12]
    tied.sort(key=lambda c: (_complexity(c["config"]), -float(c["score"])))
    members, branch = _incumbent_members(summary, sel, cands, tied, topk, honour_incumbent)
    return members, se, len(tied), branch


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
    members = [c for c in tied if c["id"] in ref_set][:topk]
    return members, "incumbent retained -> reference rule inside the reference grid"


def _reference_clf_space(adapter_path, ctx):
    return os.path.join(os.path.dirname(os.path.abspath(adapter_path)), "clf_space_%s.json" % ctx.replace("/", "_"))


def reference_deploy(data, task, ctx, clf_configs, metric="official_composite"):
    import merid.run_method as RM
    RM.SEL_METRIC = metric
    X, Xt = np.asarray(data["train_X"], float), np.asarray(data["test_X"], float)
    y = np.asarray(data["train_" + task], int)
    groups = data.get("train_subject")
    test_groups = np.array([_subject_of(i) for i in data["test_ids"]]) if data.get("test_ids") is not None else None
    yp, pool, std_oof = RM.select_clf(X, Xt, y, groups, clf_configs, task, None, ctx)
    used_decode = False
    if RM.METRIC_DECODE:
        dec_oof, dec_pred = RM._metric_decode(X, Xt, y, groups, clf_configs)
        if dec_pred is not None and dec_oof is not None and dec_oof > std_oof:
            yp, used_decode = dec_pred, True
    pooled = bool(pool) and test_groups is not None and len(test_groups) == len(yp)
    if pooled:
        yp = RM._pool_by_group(np.asarray(yp, int), test_groups)
    return np.asarray(yp, int), {"metric_decode": used_decode, "subject_pooled": pooled, "standard_oof": float(std_oof)}


def _winners(summary, topk):
    cands = [c for c in summary.get("candidates", []) if c.get("valid", True) and c.get("score") is not None]
    sel = [c for c in cands if c.get("id") == summary.get("selected_id")]
    if not sel:
        raise SystemExit("selected_id %r is not among the valid candidates" % summary.get("selected_id"))
    rest = sorted((c for c in cands if c is not sel[0]), key=lambda c: float(c["score"]), reverse=True)
    return (sel + rest)[:max(1, topk)]


def _predict(config, task, metric, X, y, groups, Xt, test_groups):
    if isinstance(config, dict) and "_decoder_strategy" in config:
        m = GroupedDecodingClassifier(config["_base_config"], task=task,
                                      strategy=config["_decoder_strategy"], metric=metric)
        m.fit(X, y, groups)
        return np.asarray(m.predict_with_groups(Xt, test_groups), int), config["_decoder_strategy"]
    m = build_model(config, int(X.shape[1]), kind="classification", task=task)
    m.fit(X, y)
    return np.asarray(m.predict(Xt), int), "plain"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--adapter", required=True, help="full-scope frozen adapter of the reference pipeline for this cell (reads the test root)")
    ap.add_argument("--ctx", required=True, help="e.g. MPDD-Young/1s")
    ap.add_argument("--task", default=None, help="binary | ternary | quinary; defaults to the summary's task")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--members", choices=["1se", "1se_pool", "top"], default="1se",
                    help="1se = the reference members when the incumbent was retained, else the 1-SE tied "
                         "set (simplest first, at most topk); 1se_pool = the 1-SE tied set of the whole pool "
                         "regardless of the incumbent (ablation); top = the topk highest development scores")
    ap.add_argument("--reference_clf_space", default=None,
                    help="frozen classifier space of the reference pipeline (default: clf_space_<ctx>.json beside --adapter); "
                         "a retained incumbent is deployed with its own select_clf/decode/pooling over this space")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)

    summary = json.load(open(a.summary, encoding="utf-8"))
    task = a.task or summary["task"]
    metric = summary.get("primary_metric") or "official_composite"
    code = open(a.adapter, encoding="utf-8").read()
    sha = hashlib.sha256(code.encode()).hexdigest()
    want = summary.get("adapter_source_sha256")
    summary["_path"] = a.summary
    if a.members in ("1se", "1se_pool"):
        winners, se, n_tied, branch = _members_1se(summary, a.topk, metric, task,
                                                   honour_incumbent=(a.members == "1se"))
        rule = "%s (SE %.4f, %d tied, %d used)" % (branch, se, n_tied, len(winners))
    else:
        winners = _winners(summary, a.topk)
        rule = "top-%d by development score" % len(winners)
    print("== MPDD-2025 %s / %s: official-test evaluation of the development winner (%s) ==" % (a.ctx, task, rule))
    print("   summary: %s  (%s, selected %s = %.4f)" % (a.summary, summary.get("protocol"), metric,
                                                       float(summary.get("primary_score") or 0)))
    print("   adapter: %s (sha %s; development staged adapter was %s - a different, training-only file by design)"
          % (a.adapter, sha[:12], (want or "?")[:12]))
    for i, c in enumerate(winners):
        print("   #%d dev %.4f  recipe=%s  config=%s" % (i + 1, float(c["score"]), c["recipe"], c["config"]))
    if a.dry_run:
        return

    t0 = time.time()
    load = _build(code)
    preds, decoders, yt, test_ids = [], [], None, None
    deployment = {"rule": "vote over members"}
    if rule.startswith("incumbent retained"):
        space_path = a.reference_clf_space or _reference_clf_space(a.adapter, a.ctx)
        clf_configs = json.load(open(space_path, encoding="utf-8"))["configs"]
        data = load(winners[0]["recipe"])
        for k in ("train_X", "train_" + task, "train_subject", "test_X", "test_" + task, "test_ids"):
            if k not in data:
                raise SystemExit("the adapter returned no %r; the full-scope reference adapter for this cell is needed" % k)
        yt = np.asarray(data["test_" + task], int)
        test_ids = list(data["test_ids"])
        yp, info = reference_deploy(data, task, a.ctx, clf_configs, metric)
        deployment = {"rule": "reference deployment (select_clf 1-SE top-3 + metric decode + subject consensus)",
                      "clf_space": space_path, **info}
        preds, decoders = [yp], ["reference:%s%s" % ("decode" if info["metric_decode"] else "vote",
                                              "+pooled" if info["subject_pooled"] else "")]
        print("   reference deployment: metric_decode=%s subject_pooled=%s (standard OOF %.4f)"
              % (info["metric_decode"], info["subject_pooled"], info["standard_oof"]), flush=True)
        winners = winners[:1]
    for c in winners if not rule.startswith("incumbent retained") else []:
        data = load(c["recipe"])
        for k in ("train_X", "train_" + task, "train_subject", "test_X", "test_" + task, "test_ids"):
            if k not in data:
                raise SystemExit("the adapter returned no %r; the full-scope reference adapter for this cell is needed" % k)
        X, y = np.asarray(data["train_X"], float), np.asarray(data["train_" + task], int)
        groups = np.asarray(data["train_subject"]).astype(str)
        Xt, yt = np.asarray(data["test_X"], float), np.asarray(data["test_" + task], int)
        test_ids = list(data["test_ids"])
        test_groups = np.array([_subject_of(i) for i in data["test_ids"]])
        yp, dec = _predict(c["config"], task, metric, X, y, groups, Xt, test_groups)
        preds.append(yp)
        decoders.append(dec)
    P = np.vstack(preds)
    yp = np.apply_along_axis(lambda col: np.bincount(col.astype(int)).argmax(), 0, P)
    acc, wacc, mf1, wf1, off = official(yt, yp)
    maj = np.full_like(yt, np.bincount(yt).argmax())
    m_acc, m_wacc, m_mf1, m_wf1, m_off = official(yt, maj)
    print("\n==================== MERID - %s / %s ====================" % (a.ctx, task))
    print("  protocol: official test, fit on train only (development winner; %s; vote over %d; decoders %s)"
          % (rule, len(winners), ",".join(sorted(set(decoders)))))
    print("  columns: Acc | Macro-F1 | W-Acc | W-F1 | TaskAcc | TaskF1 | Official")
    print("  RESULT (n=%d)  %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f"
          % (len(yt), acc, mf1, wacc, wf1, (acc + wacc) / 2, (mf1 + wf1) / 2, off))
    print("  predict-the-majority: %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f"
          % (m_acc, m_mf1, m_wacc, m_wf1, (m_acc + m_wacc) / 2, (m_mf1 + m_wf1) / 2, m_off))
    print("   done in %.0fs" % (time.time() - t0))

    out = os.path.join(os.path.dirname(os.path.abspath(a.summary)),
                       "final_test_metrics_%s%d.json" % ("1se" if a.members == "1se" else "top", len(winners)))
    json.dump({"ctx": a.ctx, "task": task, "selection_metric": metric, "n_test": int(len(yt)), "topk": len(winners),
               "member_rule": rule,
               "winners": [{"recipe": c["recipe"], "config": c["config"], "dev_score": c["score"]} for c in winners],
               "decoders": decoders, "deployment": deployment,
               "metrics": {"accuracy": acc, "macro_f1": mf1, "weighted_accuracy": wacc, "weighted_f1": wf1,
                           "task_accuracy": (acc + wacc) / 2, "task_f1": (mf1 + wf1) / 2, "official": off},
               "test_ids": [str(i) for i in test_ids] if test_ids is not None else None,
               "y_true": [int(v) for v in yt],
               "prediction": [int(v) for v in yp],
               "member_prediction": [[int(v) for v in row] for row in P],
               "majority_baseline": {"accuracy": m_acc, "macro_f1": m_mf1, "official": m_off},
               "adapter_sha256": sha, "development_adapter_sha256": want,
               "summary_sha256": hashlib.sha256(open(a.summary, "rb").read()).hexdigest(),
               "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(out, "w", encoding="utf-8"), indent=2)
    print("   metrics -> %s" % out)


if __name__ == "__main__":
    main()
