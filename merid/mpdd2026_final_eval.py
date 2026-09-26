"""Final evaluation for MPDD-2026: the heads selected on development data, scored once on the test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import Harness
from run_e2e import save_preds, score_vs_answer

PHQ_KEYS = ("audio", "video", "mode", "model", "k", "shrink", "binary_rule")
TER_KEYS = ("audio", "video", "align_mode", "gate_model", "sev_model", "sev_fusion", "k", "sev_base_mode")


def _params(entry, family, keys):
    choice = entry.get("choice") or {}
    head = choice.get(family) or {}
    p = head.get("parameters") if isinstance(head, dict) and "parameters" in head else head
    if not isinstance(p, dict) or not p:
        raise ValueError("entry has no %s parameters: %s" % (family, json.dumps(choice)[:200]))
    return {k: p[k] for k in keys if k in p}


def _entry_params(entry, family, keys):
    try:
        return _params(entry, family, keys)
    except ValueError:
        cfg = (entry.get("evaluation") or {}).get("config")
        if isinstance(cfg, dict) and cfg:
            return {k: cfg[k] for k in keys if k in cfg}
        raise


def _macro_f1(y, yp):
    from sklearn.metrics import f1_score
    return float(f1_score(y, yp, average="macro", zero_division=0))


def _ccc(y, yp):
    y, yp = np.asarray(y, float), np.asarray(yp, float)
    if len(y) < 2:
        return 0.0
    my, mp = y.mean(), yp.mean()
    vy, vp = y.var(), yp.var()
    cov = ((y - my) * (yp - mp)).mean()
    den = vy + vp + (my - mp) ** 2
    return float(2 * cov / den) if den > 0 else 0.0


def _component_scores(entry, labels):
    ev = entry.get("evaluation") or {}
    pids = [int(p) for p in ev.get("patient_ids") or []]
    if not pids:
        return None, None
    fam = entry.get("head_family")
    if fam == "binary_phq":
        oof = np.asarray(ev["phq_oof"], float)
        thr = float((ev.get("binary_rule") or {}).get("threshold", 0.0))
        yb = np.asarray([labels["binary"][p] for p in pids], int)
        yq = np.asarray([labels["phq"][p] for p in pids], float)

        def score(idx):
            if len(np.unique(yb[idx])) < 2:
                return None
            return _macro_f1(yb[idx], (oof[idx] >= thr).astype(int)) + _ccc(yq[idx], oof[idx])
    elif fam == "hierarchical_ternary":
        gate = np.asarray(ev["gate_oof"], float)[:, 1]
        sev = np.asarray(ev["severity_oof"], float)[:, 1]
        rule = ev.get("rule") or {}
        gth, sth = float(rule.get("binary_threshold", 0.5)), float(rule.get("severity_threshold", 0.5))
        pred = np.where(gate < gth, 0, np.where(sev >= sth, 2, 1))
        yt = np.asarray([labels["ternary"][p] for p in pids], int)

        def score(idx):
            if len(np.unique(yt[idx])) < 2:
                return None
            return _macro_f1(yt[idx], pred[idx])
    else:
        return None, None
    return np.arange(len(pids)), score


def _bootstrap_se(entry, labels, n_boot=1000, seed=0):
    idx, score = _component_scores(entry, labels)
    if idx is None:
        return None
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        s = score(rng.integers(0, len(idx), len(idx)))
        if s is not None:
            out.append(s)
    return float(np.std(out)) if out else None


def _normalise_reference_heads(raw):
    out = {}
    for fam, keys in (("binary_phq", PHQ_KEYS), ("hierarchical_ternary", TER_KEYS)):
        v = raw.get(fam)
        if v is None:
            continue
        items = v if isinstance(v, list) else [v]
        heads = []
        for it in items:
            p = it.get("parameters") if isinstance(it, dict) and "parameters" in it else it
            if not isinstance(p, dict):
                raise ValueError("reference head for %s is not a parameter object" % fam)
            heads.append({k: p[k] for k in keys if k in p})
        if heads:
            out[fam] = heads
    return out


def incumbent_decision(results, family, keys, ref_heads, labels, promotion_se=1.0, seed=0):
    hist = [e for e in results.get("component_history", []) if e.get("valid", True) and e.get("head_family") == family]
    want = {k: v for k, v in ref_heads[0].items() if v is not None}
    ref = None
    for e in hist:
        try:
            p = _entry_params({"choice": {family: e.get("choice") or {}}, "evaluation": e.get("evaluation")}, family, keys)
        except ValueError:
            continue
        if all(p.get(k) == v for k, v in want.items()):
            ref = e
            break
    if ref is None:
        return dict(family=family, retained=False, reason="the reference head was not evaluated in this run (no reference seed?)")
    ref_score = float(ref.get("component_score", 0.0))
    se = _bootstrap_se(ref, labels, seed=seed)
    if se is None:
        return dict(family=family, retained=False, reason="the reference head has no stored OOF to bootstrap", ref_score=ref_score)
    challengers = [(float(e.get("component_score", 0.0)), e.get("id")) for e in hist if e is not ref]
    best = max(challengers) if challengers else (None, None)
    margin = max(1e-12, promotion_se * se)
    retained = best[0] is None or best[0] <= ref_score + margin
    return dict(family=family, retained=bool(retained), ref_score=ref_score, se=se, margin=margin,
                best_challenger=best[0], best_challenger_id=best[1], promotion_se=promotion_se,
                reason=("no challenger clears the reference head by %.2f SE" % promotion_se) if retained else
                       ("a challenger clears the reference head by more than %.2f SE" % promotion_se))


def candidates(results, topk, reference_heads=None, labels=None, promotion_se=1.0, seed=0):
    best = results["best"]
    phq = [_params(best, "binary_phq", PHQ_KEYS)]
    ter = [_params(best, "hierarchical_ternary", TER_KEYS)]
    meta = {"winner_dev_track": best.get("track"), "phq_scores": [best.get("track")], "ter_scores": [best.get("track")]}
    retained = {}
    if reference_heads:
        if labels is None:
            raise ValueError("the incumbent rule needs development labels")
        meta["incumbent"] = {}
        for fam, keys in (("binary_phq", PHQ_KEYS), ("hierarchical_ternary", TER_KEYS)):
            if fam not in reference_heads:
                continue
            d = incumbent_decision(results, fam, keys, reference_heads[fam], labels, promotion_se, seed)
            meta["incumbent"][fam] = d
            if d["retained"]:
                retained[fam] = list(reference_heads[fam])[:max(1, topk)]
        if "binary_phq" in retained:
            phq[:] = retained["binary_phq"]
            meta["phq_scores"] = [meta["incumbent"]["binary_phq"].get("ref_score")] * len(phq)
        if "hierarchical_ternary" in retained:
            ter[:] = retained["hierarchical_ternary"]
            meta["ter_scores"] = [meta["incumbent"]["hierarchical_ternary"].get("ref_score")] * len(ter)
    if topk > 1:
        hist = [e for e in results.get("component_history", []) if e.get("valid", True)]

        def score(e):
            cs = e.get("component_score")
            if isinstance(cs, (int, float)):
                return float(cs)
            ev = e.get("evaluation") or {}
            return float(ev.get("terF1", 0.0)) if e.get("head_family") == "hierarchical_ternary"                 else float(ev.get("binF1", 0.0)) + float(ev.get("ccc", 0.0))
        for fam, keys, lst, sk in (("binary_phq", PHQ_KEYS, phq, "phq_scores"),
                                   ("hierarchical_ternary", TER_KEYS, ter, "ter_scores")):
            if fam in retained:
                continue
            seen = {json.dumps(lst[0], sort_keys=True)}
            for e in sorted((e for e in hist if e.get("head_family") == fam), key=score, reverse=True):
                try:
                    p = _entry_params({"choice": {fam: e.get("choice") or {}}, "evaluation": e.get("evaluation")},
                                      fam, keys)
                except ValueError:
                    continue
                key = json.dumps(p, sort_keys=True)
                if key in seen or len(lst) >= topk:
                    continue
                seen.add(key)
                lst.append(p)
                meta[sk].append(score(e))
    return phq, ter, meta


def _zip_from_csv(path):
    import csv
    import tempfile
    import zipfile
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    idc = next(k for k in rows[0] if k.strip().lower() == "id")
    phq = next(k for k in rows[0] if "phq" in k.lower())
    nl = chr(10)
    b = "id,binary_pred,phq9_pred" + nl + "".join("%s,%s,%s" % (r[idc], int(float(r["label2"])), r[phq]) + nl for r in rows)
    t = "id,ternary_pred" + nl + "".join("%s,%s" % (r[idc], int(float(r["label3"]))) + nl for r in rows)
    fh = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(fh, "w") as z:
        z.writestr("binary.csv", b)
        z.writestr("ternary.csv", t)
    fh.close()
    return fh.name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="development results.json")
    ap.add_argument("--track", default="Track2", choices=["Track1", "Track2"])
    ap.add_argument("--split_root", required=True)
    ap.add_argument("--answer_zip", default=None, help="answer key in submission-zip form (binary.csv + ternary.csv)")
    ap.add_argument("--answer_csv", default=None,
                    help="answer key as the official split_labels_test.csv (ID,label2,label3,PHQ-9); "
                         "scoring ONLY - either this or --answer_zip")
    ap.add_argument("--topk", type=int, default=1,
                    help="1 = the selected winner alone; 5 = the run_e2e-style ensemble of the top-5 development ranking")
    ap.add_argument("--save_dir", default=None, help="also dump binary.csv/ternary.csv in the answer-key schema")
    ap.add_argument("--reference_heads", default=None,
                    help="JSON {family: [head params...]} of the reference heads (best first): the incumbent rule keeps "
                         "them unless a challenger beats the first reference head by --promotion_se SEs on the development OOF")
    ap.add_argument("--promotion_se", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed for the incumbent rule")
    ap.add_argument("--dry_run", action="store_true", help="print the resolved configurations and stop")
    a = ap.parse_args(argv)
    if bool(a.answer_zip) == bool(a.answer_csv):
        ap.error("give exactly one of --answer_zip / --answer_csv")

    results = json.load(open(a.results, encoding="utf-8"))
    if results.get("evaluation") != "development":
        raise SystemExit("refusing: %s is not a development result (evaluation=%r)"
                         % (a.results, results.get("evaluation")))
    reference_heads = None
    h = None
    if a.reference_heads:
        reference_heads = _normalise_reference_heads(json.load(open(a.reference_heads, encoding="utf-8-sig")))
        h = Harness(track=a.track, split_root=a.split_root)
        labels = {"binary": h.labels["binary"], "ternary": h.labels["ternary"], "phq": h.phq}
        phq_list, ter_list, meta = candidates(results, a.topk, reference_heads, labels, a.promotion_se, a.seed)
        for fam, d in meta.get("incumbent", {}).items():
            print("   [incumbent %s] %s: %s (reference %.4f, SE %.4f, best challenger %s)" % (
                fam, "RETAINED" if d["retained"] else "PROMOTED", d.get("reason"), d.get("ref_score") or 0,
                d.get("se") or 0, ("%.4f" % d["best_challenger"]) if d.get("best_challenger") is not None else "-"))
    else:
        phq_list, ter_list, meta = candidates(results, a.topk)
    print("== MPDD-2026 %s: final test evaluation of the development winner (PHQ top-%d blend, ternary top-%d vote) =="
          % (a.track, len(phq_list), len(ter_list)))
    print("   results: %s  (best dev track %.4f, protocol %s)" % (
        a.results, float(results["best"].get("track", float("nan"))),
        str(results.get("protocol_fingerprint", "?"))[:16]))
    for i, p in enumerate(phq_list):
        print("   PHQ #%d (%.4f) %s" % (i + 1, float(meta["phq_scores"][i] or 0), p))
    for i, p in enumerate(ter_list):
        print("   TER #%d (%.4f) %s" % (i + 1, float(meta["ter_scores"][i] or 0), p))
    if a.dry_run:
        return

    t0 = time.time()
    if h is None:
        h = Harness(track=a.track, split_root=a.split_root)
    phq_preds = [h.predict_phq_binary(p) for p in phq_list]
    ter_preds = [h.predict_ternary(p) for p in ter_list]
    ids = list(phq_preds[0])
    preds = {}
    for pid in ids:
        preds[pid] = {"phq": float(np.mean([p[pid]["phq"] for p in phq_preds])),
                      "binary": int(np.bincount([p[pid]["binary"] for p in phq_preds]).argmax()),
                      "ternary": int(np.bincount([tp.get(pid, 1) for tp in ter_preds]).argmax())}
    if a.save_dir:
        save_preds(preds, a.save_dir)
        print("   saved predictions -> %s" % a.save_dir)
    s = score_vs_answer(preds, a.answer_zip or _zip_from_csv(a.answer_csv))
    print("   scored %d test ids in %.0fs" % (s["n"], time.time() - t0))
    print("\n  Task     | Accuracy | Macro-F1 | Kappa")
    print("  Binary   |  %.4f |  %.4f | %+.4f" % (s["binAcc"], s["binF1"], s["binK"]))
    print("  Ternary  |  %.4f |  %.4f | %+.4f" % (s["terAcc"], s["terF1"], s["terK"]))
    print("  PHQ-9    | RMSE %.4f | MAE %.4f | CCC %+.4f | R2 %+.4f" % (s["rmse"], s["mae"], s["ccc"], s["r2"]))
    print("\n  === test TRACK = (binF1+terF1+CCC)/3 = %.4f ===" % s["track"])

    out = os.path.join(os.path.dirname(os.path.abspath(a.results)), "final_test_metrics_top%d.json" % a.topk)
    record = {"track": a.track, "split_root": a.split_root, "topk": a.topk,
              "member_rule": ("incumbent rule: " + ", ".join("%s %s" % (f, "retained" if d["retained"] else "promoted")
                                                            for f, d in meta.get("incumbent", {}).items()))
              if meta.get("incumbent") else "top-%d by development score" % a.topk,
              "reference_heads": a.reference_heads, "promotion_se": a.promotion_se if a.reference_heads else None,
              "phq_heads": phq_list, "ternary_heads": ter_list, "dev_scores": meta,
              "metrics": s, "results_sha256": hashlib.sha256(open(a.results, "rb").read()).hexdigest(),
              "answer_key": os.path.basename(a.answer_zip or a.answer_csv), "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(record, open(out, "w", encoding="utf-8"), indent=2)
    print("   metrics -> %s" % out)


if __name__ == "__main__":
    main()
