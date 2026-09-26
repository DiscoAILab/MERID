"""Measures whether an input column determines the label; the result enters the inspection report as target-defining evidence."""
from __future__ import annotations

import argparse
import json

import numpy as np


def _thresholds(x, y_codes, max_leaves):
    from sklearn.tree import DecisionTreeClassifier
    t = DecisionTreeClassifier(max_leaf_nodes=max_leaves, random_state=0)
    t.fit(np.asarray(x, float).reshape(-1, 1), y_codes)
    acc = float(t.score(np.asarray(x, float).reshape(-1, 1), y_codes))
    cuts = sorted(float(v) for v, f in zip(t.tree_.threshold, t.tree_.feature) if f >= 0)
    return acc, cuts


def analyse(values, labels, columns, exact=0.999, strong=0.95):
    order = sorted(set(map(str, labels)))
    code = {v: i for i, v in enumerate(order)}
    y = np.asarray([code[str(v)] for v in labels], int)
    n_classes = len(order)
    out = []
    for col in columns:
        x = np.asarray(values[col], float)
        if not np.isfinite(x).all() or len(np.unique(x)) < 2:
            continue
        pairs = {}
        collisions = 0
        for xi, yi in zip(x, y):
            seen = pairs.setdefault(xi, set())
            seen.add(int(yi))
            if len(seen) > 1:
                collisions += 1
        functional = 1.0 - (sum(1 for s in pairs.values() if len(s) > 1) / max(1, len(pairs)))
        acc, cuts = _thresholds(x, y, n_classes)
        wide, _ = _thresholds(x, y, n_classes + 3)
        rec = {"column": col, "threshold_accuracy": acc, "accuracy_with_extra_cuts": wide,
               "value_is_functional": functional, "cuts": cuts, "n_distinct": int(len(np.unique(x)))}
        rec["verdict"] = ("label is a threshold function of this column" if acc >= exact else
                          "label is almost a threshold function of this column" if acc >= strong else
                          "not a threshold function")
        out.append(rec)
    out.sort(key=lambda r: -r["threshold_accuracy"])
    return out, order


def analyse_groups(values, labels, columns, top_singles=4, max_size=3):
    from itertools import combinations
    from sklearn.tree import DecisionTreeClassifier
    order = sorted(set(map(str, labels)))
    code = {v: i for i, v in enumerate(order)}
    y = np.asarray([code[str(v)] for v in labels], int)
    n_classes = len(order)
    singles, _ = analyse(values, labels, columns)
    cands = [r["column"] for r in singles[:top_singles]]
    out = []
    for size in range(2, max_size + 1):
        for combo in combinations(cands, size):
            X = np.column_stack([np.asarray(values[c], float) for c in combo])
            t = DecisionTreeClassifier(max_leaf_nodes=max(n_classes, 2) * 4, random_state=0).fit(X, y)
            joint = float(t.score(X, y))
            total = X.sum(axis=1)
            acc_sum, cuts = _thresholds(total, y, n_classes)
            out.append({"columns": list(combo), "joint_rule_accuracy": joint,
                        "sum_threshold_accuracy": acc_sum, "sum_cuts": cuts})
    out.sort(key=lambda r: -max(r["joint_rule_accuracy"], r["sum_threshold_accuracy"]))
    return out, order


def report_groups(groups, exact=0.999, strong=0.95, top=5):
    lines = ["COMBINATIONS (training rows only)"]
    for g in groups[:top]:
        lines.append("  %-44s joint rule %.4f | sum thresholded %.4f at %s"
                     % ("+".join(g["columns"]), g["joint_rule_accuracy"], g["sum_threshold_accuracy"],
                        ", ".join("%.4g" % c for c in g["sum_cuts"]) or "none"))
    best = max((max(g["joint_rule_accuracy"], g["sum_threshold_accuracy"]) for g in groups), default=0.0)
    if best >= strong:
        g = max(groups, key=lambda g: max(g["joint_rule_accuracy"], g["sum_threshold_accuracy"]))
        lines.append("  PROVENANCE: %s reproduce the label (%.4f). That combination is the formula the "
                     "label came from." % ("+".join(g["columns"]), best))
    else:
        lines.append("  No combination of the strongest columns reproduces the label either (best %.4f): "
                     "they predict it well without defining it, which is what evidence looks like." % best)
    return chr(10).join(lines)


def report(records, order, exact=0.999, strong=0.95, top=6):
    lines = ["LABEL PROVENANCE (training rows only; %d classes: %s)" % (len(order), ", ".join(map(str, order)))]
    hits = [r for r in records if r["threshold_accuracy"] >= strong]
    for r in records[:top]:
        lines.append("  %-28s threshold-rule accuracy %.4f (with spare cuts %.4f); cuts at %s"
                     % (r["column"], r["threshold_accuracy"], r["accuracy_with_extra_cuts"],
                        ", ".join("%.4g" % c for c in r["cuts"]) or "none"))
    if not hits:
        lines.append("  No column reproduces the label by thresholding: on this evidence no single column "
                     "is the instrument the label was derived from.")
    else:
        best = hits[0]
        lines.append("  PROVENANCE: the label is reproduced from %r alone by thresholding at %s "
                     "(accuracy %.4f). The instrument/item/formula is therefore that column and those "
                     "cut points, and a model using it is recovering the label rather than predicting it."
                     % (best["column"], ", ".join("%.4g" % c for c in best["cuts"]), best["threshold_accuracy"]))
        if len(hits) > 1:
            lines.append("  Same finding for: %s" % ", ".join(r["column"] for r in hits[1:]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--split_csv", help="optional row_index,split file; only rows marked train are used")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import csv as _csv
    rows = list(_csv.DictReader(open(a.csv, encoding="utf-8-sig")))
    if a.split_csv:
        keep = {int(r["row_index"]) for r in _csv.DictReader(open(a.split_csv, encoding="utf-8-sig"))
                if str(r.get("split", "")).strip().lower() in ("train", "")}
        rows = [r for i, r in enumerate(rows) if i in keep]
    if not rows:
        raise SystemExit("no rows")
    labels = [r[a.label] for r in rows]
    columns, values = [], {}
    for col in rows[0]:
        if col == a.label:
            continue
        try:
            values[col] = [float(r[col]) for r in rows]
        except (TypeError, ValueError):
            continue
        columns.append(col)
    records, order = analyse(values, labels, columns)
    text = report(records, order)
    groups, _ = analyse_groups(values, labels, columns)
    gtext = report_groups(groups)
    print("   %d training rows, %d numeric columns" % (len(rows), len(columns)))
    print(text)
    print(gtext)
    if a.out:
        json.dump({"csv": a.csv, "label": a.label, "n_rows": len(rows), "classes": order,
                   "columns": records, "combinations": groups, "report": text + chr(10) + gtext},
                  open(a.out, "w", encoding="utf-8"), indent=2)
        print("   -> %s" % a.out)


if __name__ == "__main__":
    main()
