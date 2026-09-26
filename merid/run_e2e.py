"""MPDD-2026 end-to-end run and the test scoring helpers."""
import argparse
import csv
import io
import os
import sys
import zipfile

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import Harness
from manager import optimize
from profiler import profile as build_profile
from proposer import ClaudeProposer, RandomProposer
from space import TERNARY_SPACE


def build_proposer(args):
    if args.proposer == "random":
        return RandomProposer(seed=args.seed)
    return ClaudeProposer(model=args.llm_model or "claude-opus-4-8", provider="anthropic",
                          base_url=args.llm_base_url, api_key=args.llm_api_key)


def ternary_grid():
    a, v = "opensmile", "openface"
    combos = []
    for am in ["alignment", "alignment_ext"]:
        for gm in ["svcC0.1", "logregC0.1", "svcC1.0"]:
            for sm in ["gb0.05d1", "gb0.03d1", "svcC0.1"]:
                for k in (64, 128):
                    combos.append({"audio": a, "video": v, "align_mode": am, "gate_model": gm,
                                   "sev_model": sm, "sev_fusion": None, "k": k})
    return combos


def _ccc(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    d = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    return float(2 * cov / d) if d > 1e-10 else 0.0


def score_vs_answer(preds, answer_zip):
    with zipfile.ZipFile(answer_zip) as zf:
        names = zf.namelist()
        b = list(csv.DictReader(io.StringIO(zf.read([n for n in names if n.endswith("binary.csv")][0]).decode("utf-8-sig"))))
        t = list(csv.DictReader(io.StringIO(zf.read([n for n in names if n.endswith("ternary.csv")][0]).decode("utf-8-sig"))))
    tb = {int(r["id"]): int(float(r["binary_pred"])) for r in b}
    tt = {int(r["id"]): int(float(r["ternary_pred"])) for r in t}
    tp = {int(r["id"]): float(r["phq9_pred"]) for r in b}
    ids = sorted(pid for pid in preds if pid in tb)
    yb = [tb[i] for i in ids]; qb = [preds[i]["binary"] for i in ids]
    yt = [tt[i] for i in ids]; qt = [preds[i]["ternary"] for i in ids]
    gp = np.asarray([tp[i] for i in ids], float); pp = np.asarray([preds[i]["phq"] for i in ids], float)
    binF1 = f1_score(yb, qb, average="macro", zero_division=0)
    terF1 = f1_score(yt, qt, average="macro", zero_division=0)
    ccc = _ccc(gp, pp)
    rmse = float(np.sqrt(((gp - pp) ** 2).mean())); mae = float(np.abs(gp - pp).mean())
    ss = float(((gp - gp.mean()) ** 2).sum())
    r2 = float(1 - ((gp - pp) ** 2).sum() / ss) if ss > 1e-12 else 0.0
    return {"n": len(ids), "binAcc": accuracy_score(yb, qb), "binF1": binF1, "binK": cohen_kappa_score(yb, qb),
            "terAcc": accuracy_score(yt, qt), "terF1": terF1, "terK": cohen_kappa_score(yt, qt),
            "rmse": rmse, "mae": mae, "ccc": ccc, "r2": r2, "track": (binF1 + terF1 + ccc) / 3.0}


def save_preds(preds, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ids = sorted(preds)
    with open(os.path.join(save_dir, "binary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "binary_pred", "phq9_pred"])
        for i in ids:
            w.writerow([i, preds[i]["binary"], preds[i]["phq"]])
    with open(os.path.join(save_dir, "ternary.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "ternary_pred"])
        for i in ids:
            w.writerow([i, preds[i]["ternary"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="Track2", choices=["Track1", "Track2"])
    ap.add_argument("--split_root", default="<SPLIT_ROOT>",
                    help="directory of the fold split CSVs for the chosen track")
    ap.add_argument("--proposer", default="anthropic", choices=["anthropic", "random"])
    ap.add_argument("--budget", type=int, default=10)
    ap.add_argument("--topk", type=int, default=5, help="ternary: majority-vote the top-K OOF configs (robust vs OOF-argmax)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm_model", default=None)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--llm_api_key", default=None)
    ap.add_argument("--answer_zip", required=True, help="eval ONLY — never used in selection")
    ap.add_argument("--save_dir", default=None, help="optional: dump binary.csv+ternary.csv here (answer-key schema)")
    a = ap.parse_args()

    h = Harness(track=a.track, split_root=a.split_root)
    prof = build_profile(h)

    print("== SELECT PHQ+binary head (agent searches OOF track; ROBUST top-K) ==", flush=True)
    history, best = optimize(h, build_proposer(a), budget=a.budget, seed=a.seed, profile=prof)
    phq_ranked = sorted([x for x in history if x.get("track") is not None], key=lambda x: x["track"], reverse=True)
    phq_topk = [x["choice"] for x in phq_ranked[:a.topk]]
    print(f"  top-{a.topk} PHQ by OOF track: " + ", ".join(f"{x['track']:.4f}" for x in phq_ranked[:a.topk]), flush=True)

    print("\n== SELECT ternary head (agent searches OOF terF1; ROBUST top-K) ==", flush=True)
    scored = sorted(((h.evaluate_ternary(t)["terF1"], t) for t in ternary_grid()), key=lambda x: x[0], reverse=True)
    ter_topk = [t for _, t in scored[:a.topk]]
    print(f"  top-{a.topk} ternary by OOF terF1: " + ", ".join(f"{s:.4f}" for s, _ in scored[:a.topk]), flush=True)
    print(f"  (single OOF-argmax overfits n~88; top-{a.topk} ENSEMBLE both heads = robust, decided on OOF)", flush=True)

    print("\n== PREDICT test + SCORE vs answer (eval only) ==", flush=True)
    phq_preds = [h.predict_phq_binary(c) for c in phq_topk]
    ter_preds = [h.predict_ternary(t) for t in ter_topk]
    ids = list(phq_preds[0])
    preds = {}
    for pid in ids:
        phq = float(np.mean([p[pid]["phq"] for p in phq_preds]))
        binary = int(np.bincount([p[pid]["binary"] for p in phq_preds]).argmax())
        ternary = int(np.bincount([tp.get(pid, 1) for tp in ter_preds]).argmax())
        preds[pid] = {"binary": binary, "ternary": ternary, "phq": phq}
    if a.save_dir:
        save_preds(preds, a.save_dir)
        print(f"  saved predictions -> {a.save_dir}/binary.csv + ternary.csv")
    s = score_vs_answer(preds, a.answer_zip)
    print(f"  scored {s['n']} test ids")
    print("\n  Task     | Accuracy | Macro-F1 | Kappa")
    print(f"  Binary   |  {s['binAcc']:.4f} |  {s['binF1']:.4f} | {s['binK']:+.4f}")
    print(f"  Ternary  |  {s['terAcc']:.4f} |  {s['terF1']:.4f} | {s['terK']:+.4f}")
    print(f"  PHQ-9    | RMSE {s['rmse']:.4f} | MAE {s['mae']:.4f} | CCC {s['ccc']:+.4f} | R2 {s['r2']:+.4f}")
    print(f"\n  === MERID test TRACK = (binF1+terF1+CCC)/3 = {s['track']:.4f} ===")


if __name__ == "__main__":
    main()
