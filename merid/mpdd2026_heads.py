"""MPDD-2026 candidate heads, their calibration, PHQ-9 threshold rules and metrics."""
from __future__ import annotations

import math
import warnings
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from mpdd2026_data import create_train_val_split


def safe_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(out) or math.isinf(out) else out


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    if not labels:
        return 0.0
    vals = []
    for label in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == label and b == label)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != label and b == label)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == label and b != label)
        denom = 2 * tp + fp + fn
        vals.append((2 * tp / denom) if denom else 0.0)
    return safe_float(sum(vals) / len(vals))


def cohen_kappa(y_true: list[int], y_pred: list[int]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    labels = sorted(set(y_true) | set(y_pred))
    observed = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    tc = Counter(y_true)
    pc = Counter(y_pred)
    expected = sum((tc[label] / n) * (pc[label] / n) for label in labels)
    denom = 1 - expected
    return safe_float((observed - expected) / denom) if denom > 1e-10 else 0.0


def sweep_binary_threshold(oof_probs: dict[int, np.ndarray], labels: dict[int, int], grid: list[float]) -> dict[str, Any]:
    ids = sorted(oof_probs)
    y_true = [labels[pid] for pid in ids]
    rows = []
    for threshold in grid:
        y_pred = [1 if float(oof_probs[pid][1]) >= threshold else 0 for pid in ids]
        f1 = macro_f1(y_true, y_pred)
        kappa = cohen_kappa(y_true, y_pred)
        rows.append(
            {
                "threshold": threshold,
                "f1": f1,
                "kappa": kappa,
                "selection": (f1 + kappa) / 2,
                "dist": dict(Counter(y_pred)),
            }
        )
    rows.sort(key=lambda row: (row["selection"], row["f1"], -abs(float(row["threshold"]) - 0.5)), reverse=True)
    return {"best": rows[0], "rows": rows}


def score_selection(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    f1 = macro_f1(y_true, y_pred)
    kappa = cohen_kappa(y_true, y_pred)
    return {
        "f1": f1,
        "kappa": kappa,
        "selection": (f1 + kappa) / 2,
        "dist": dict(Counter(y_pred)),
    }


def threshold_grid(spec: str) -> list[float]:
    start_s, stop_s, step_s = spec.split(":")
    start = float(start_s)
    stop = float(stop_s)
    step = float(step_s)
    vals = []
    cur = start
    while cur <= stop + 1e-12:
        vals.append(round(cur, 6))
        cur += step
    return vals


def apply_ternary_rule(probs: np.ndarray, rule: dict[str, Any]) -> int:
    family = str(rule["family"])
    if family == "argmax":
        return int(np.argmax(probs))
    if family == "gate_nonzero":
        t1 = float(rule["t1"])
        t2 = float(rule["t2"])
        p1 = float(probs[1])
        p2 = float(probs[2])
        if p2 >= t2 and p2 >= p1:
            return 2
        if p1 >= t1:
            return 1
        if p2 >= t2:
            return 2
        return 0
    if family == "logprob_bias":
        eps = 1e-12
        b1 = float(rule["b1"])
        b2 = float(rule["b2"])
        scores = np.log(np.clip(probs, eps, 1.0)) + np.asarray([0.0, b1, b2], dtype=np.float64)
        return int(np.argmax(scores))
    raise ValueError(f"unknown ternary rule family: {family}")


def sweep_ternary_rule(
    oof_probs: dict[int, np.ndarray],
    labels: dict[int, int],
    prob_grid: list[float],
    bias_grid: list[float],
) -> dict[str, Any]:
    ids = sorted(oof_probs)
    y_true = [labels[pid] for pid in ids]
    rows: list[dict[str, Any]] = []

    argmax_rule = {"family": "argmax"}
    y_pred = [apply_ternary_rule(oof_probs[pid], argmax_rule) for pid in ids]
    rows.append({**argmax_rule, **score_selection(y_true, y_pred), "complexity": 0.0})

    for t1 in prob_grid:
        for t2 in prob_grid:
            rule = {"family": "gate_nonzero", "t1": t1, "t2": t2}
            y_pred = [apply_ternary_rule(oof_probs[pid], rule) for pid in ids]
            rows.append(
                {
                    **rule,
                    **score_selection(y_true, y_pred),
                    "complexity": abs(t1 - 0.5) + abs(t2 - 0.5) + 0.05,
                }
            )

    for b1 in bias_grid:
        for b2 in bias_grid:
            rule = {"family": "logprob_bias", "b1": b1, "b2": b2}
            y_pred = [apply_ternary_rule(oof_probs[pid], rule) for pid in ids]
            rows.append(
                {
                    **rule,
                    **score_selection(y_true, y_pred),
                    "complexity": abs(b1) + abs(b2) + 0.10,
                }
            )

    rows.sort(
        key=lambda row: (
            float(row["selection"]),
            float(row["f1"]),
            float(row["kappa"]),
            -float(row["complexity"]),
        ),
        reverse=True,
    )
    return {"best": rows[0], "rows": rows}


def compact_rule(rule: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if "family" in rule:
        out["family"] = rule["family"]
    for key in ("threshold", "t1", "t2", "b1", "b2", "f1", "kappa", "selection", "dist"):
        if key in rule:
            value = rule[key]
            if isinstance(value, float):
                out[key] = round(value, 6)
            else:
                out[key] = value
    return out


def _patient_id(value) -> str:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("patient IDs must be finite integral values") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("patient IDs must be finite integral values")
    return str(int(number))


def _class_label(value) -> int:
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("classification labels must be finite integral values") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError("classification labels must be finite integral values")
    return int(number)


def validate_classification_labels(labels, source_ids, num_classes):
    normalized_source = [_patient_id(value) for value in source_ids]
    label_key_pairs = [(_patient_id(pid), pid) for pid in labels.keys()]
    if len(label_key_pairs) != len({pid for pid, _ in label_key_pairs}):
        raise ValueError("classification label patient IDs must be unique")
    original_label_keys = dict(label_key_pairs)
    if set(original_label_keys) != set(normalized_source):
        raise ValueError("classification labels must exactly cover protocol patients")
    normalized = {pid: _class_label(labels[original_label_keys[pid]]) for pid in normalized_source}
    if any(value < 0 or value >= num_classes for value in normalized.values()):
        raise ValueError("classification labels are outside the requested class range")
    return {original: normalized[normalized_id]
            for original, normalized_id in zip(source_ids, normalized_source)}


def evaluate_development_probabilities(
    bank,
    labels,
    *,
    fold_protocol,
    model_name,
    k,
    seeds,
    num_classes,
    make_classifier,
    predict_proba_full,
    train_subset_ids=None,
    required_train_classes=None,
    seed_offset_by_fold=False,
):
    # Package import first, so the caller and this module share one DevelopmentFoldProtocol class.
    try:
        from merid.mpdd2026_protocol import iter_development_folds
    except ModuleNotFoundError as exc:
        if exc.name not in ("merid", "merid.mpdd2026_protocol"):
            raise
        from mpdd2026_protocol import iter_development_folds

    fit_calls = 0
    completed_folds = 0
    try:
        if fold_protocol is None:
            raise ValueError("development classification requires fold_protocol")
        if bank.get("data_scope") != "development":
            raise ValueError("development classification requires a development bank")
        if "test_ids" in bank or "test_x" in bank:
            raise ValueError("development classification bank must not contain test data")
        source_ids = list(bank["train_ids"])
        x = np.asarray(bank["train_x"], dtype=float)
        if x.ndim != 2 or x.shape[0] != len(source_ids) or x.shape[1] == 0:
            raise ValueError("development classification requires an aligned 2D feature matrix")
        if not np.isfinite(x).all():
            raise ValueError("development classification features must be finite")
        normalized_source = [_patient_id(value) for value in source_ids]
        if len(normalized_source) != len(set(normalized_source)):
            raise ValueError("development classification patient IDs must be unique")
        validated_labels = validate_classification_labels(labels, source_ids, num_classes)
        normalized_labels = {_patient_id(pid): value for pid, value in validated_labels.items()}
        subset = None if train_subset_ids is None else [_patient_id(pid) for pid in train_subset_ids]
        seed_values = tuple(seeds)
        if not seed_values:
            raise ValueError("development classification requires at least one seed")
        folds = iter_development_folds(
            fold_protocol, source_ids, train_subset_ids=subset, validation_subset_ids=None)
        oof_lists = {pid: [] for pid in normalized_source}
        for view in folds:
            y_train = np.asarray([normalized_labels[pid] for pid in view.train_ids], dtype=int)
            if required_train_classes is not None and set(y_train.tolist()) != set(required_train_classes):
                raise ValueError("severity candidate needs both class1 and class2 in every outer training fold")
            for seed in seed_values:
                effective_seed = int(seed) + view.fold if seed_offset_by_fold else int(seed)
                model = make_classifier(model_name, min(k, x.shape[1]), x.shape[1], effective_seed)
                fit_calls += 1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(x[list(view.train_indices)], y_train)
                query_x = x[list(view.validation_indices)]
                raw_probabilities = np.asarray(model.predict_proba(query_x), dtype=float)
                classes = np.asarray(getattr(model, "classes_", None))
                if raw_probabilities.ndim != 2 or raw_probabilities.shape[0] != len(view.validation_indices):
                    raise ValueError("classifier probabilities must be finite and have exact query shape")
                if (classes.ndim != 1 or len(classes) != raw_probabilities.shape[1]
                        or not np.isfinite(classes.astype(float)).all()
                        or np.any(classes.astype(float) != np.floor(classes.astype(float)))):
                    raise ValueError("classifier classes must be finite unique integral values")
                class_ids = classes.astype(int)
                if len(set(class_ids.tolist())) != len(class_ids) or np.any(class_ids < 0) or np.any(class_ids >= num_classes):
                    raise ValueError("classifier classes must be finite unique integral values in range")
                if (not np.isfinite(raw_probabilities).all() or np.any(raw_probabilities < 0)
                        or np.any(raw_probabilities > 1)
                        or not np.allclose(raw_probabilities.sum(axis=1), 1., atol=1e-8, rtol=0)):
                    raise ValueError("raw classifier probabilities must be finite normalized rows in [0, 1]")
                probabilities = np.zeros((len(query_x), num_classes), dtype=float)
                for column, class_id in enumerate(class_ids):
                    probabilities[:, class_id] = raw_probabilities[:, column]
                expected = (len(view.validation_indices), num_classes)
                if probabilities.shape != expected or not np.isfinite(probabilities).all():
                    raise ValueError("classifier probabilities must be finite and have exact query shape")
                if np.any(probabilities < 0) or np.any(probabilities > 1):
                    raise ValueError("classifier probabilities must lie in [0, 1]")
                if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0):
                    raise ValueError("classifier probability rows must sum to one")
                for pid, probability in zip(view.validation_ids, probabilities):
                    oof_lists[pid].append(probability)
            completed_folds += 1
        expected_predictions = len(seed_values)
        if any(len(values) != expected_predictions for values in oof_lists.values()):
            raise ValueError("development OOF predictions do not cover every patient exactly once per seed")
        oof_by_normalized = {pid: np.mean(values, axis=0) for pid, values in oof_lists.items()}
        oof = {original: oof_by_normalized[normalized]
               for original, normalized in zip(source_ids, normalized_source)}
        return {
            "patient_ids": source_ids,
            "oof": oof,
            "fit_calls": fit_calls,
            "completed_folds": completed_folds,
            "n_protocols": 1,
            "data_scope": "development",
            "protocol_fingerprint": fold_protocol.protocol_fingerprint,
            "protocol_evidence_fingerprint": fold_protocol.evidence_fingerprint,
            "source_ids_fingerprint": folds[0].source_ids_fingerprint,
        }
    except Exception as exc:
        exc.fit_calls = fit_calls
        exc.completed_folds = completed_folds
        raise


def safe_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    mx = float(y_true.mean())
    my = float(y_pred.mean())
    vx = float(y_true.var())
    vy = float(y_pred.var())
    cov = float(((y_true - mx) * (y_pred - my)).mean())
    denom = vx + vy + (mx - my) ** 2
    return float((2.0 * cov / denom) if denom > 1e-12 else 0.0)


def phq_log1p_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.log1p(np.clip(values, 0.0, None))


def phq_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_log = phq_log1p_array(y_true)
    y_pred_log = phq_log1p_array(y_pred)
    return {
        "ccc": safe_ccc(y_true_log, y_pred_log),
        "rmse": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae": float(mean_absolute_error(y_true_log, y_pred_log)),
    }


def make_classifier(model_name: str, k: int, n_features: int, seed: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    use_pca = model_name.startswith("pca")
    base_name = model_name[3:] if use_pca else model_name
    if use_pca:
        steps.append(("pca", PCA(n_components=min(k, n_features), whiten=True, random_state=seed)))
    elif k > 0 and k < n_features:
        steps.append(("select", SelectKBest(f_classif, k=k)))
    if base_name.startswith("logreg"):
        c = float(base_name.split("C", 1)[1])
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
    elif base_name.startswith("svcrbf"):
        c = float(base_name.split("C", 1)[1])
        model = SVC(C=c, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    elif base_name.startswith("svc"):
        c = float(base_name.split("C", 1)[1])
        model = SVC(C=c, kernel="linear", class_weight="balanced", probability=True, random_state=seed)
    elif base_name == "lda":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif base_name == "gnb":
        model = GaussianNB()
    elif base_name.startswith("rf"):
        depth_s = base_name.split("d", 1)[1]
        depth = None if depth_s == "none" else int(depth_s)
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    elif base_name.startswith("et"):
        depth_s = base_name.split("d", 1)[1]
        depth = None if depth_s == "none" else int(depth_s)
        model = ExtraTreesClassifier(
            n_estimators=600,
            max_depth=depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    elif base_name.startswith("gb"):
        parts = base_name.replace("gb", "").split("d")
        learning_rate = float(parts[0])
        depth = int(parts[1])
        model = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=learning_rate,
            max_depth=depth,
            min_samples_leaf=2,
            random_state=seed,
        )
    elif base_name.startswith("knn"):
        if base_name.startswith("knnk"):
            n_neighbors = int(base_name.split("knnk", 1)[1])
        else:
            n_neighbors = int(base_name.split("knn", 1)[1])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    elif base_name.startswith("qda"):
        reg_param = float(base_name.split("r", 1)[1])
        model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
    else:
        raise ValueError(f"Unknown classifier: {model_name}")
    steps.append(("model", model))
    return Pipeline(steps)


def predict_proba_full(model: Pipeline, x: np.ndarray, num_classes: int) -> np.ndarray:
    probs = model.predict_proba(x)
    out = np.zeros((x.shape[0], num_classes), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        cls_i = int(cls)
        if 0 <= cls_i < num_classes:
            out[:, cls_i] = probs[:, col]
    row_sum = out.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 1e-12] = 1.0
    return out / row_sum


def evaluate_classifier_candidate(
    bank: dict[str, Any],
    labels: dict[int, int],
    split_root: Path,
    task: str,
    model_name: str,
    k: int,
    seeds: tuple[int, ...],
    *,
    data_scope: str = "all",
    fold_protocol=None,
) -> dict[str, Any]:
    if data_scope not in ("all", "development"):
        raise ValueError("data_scope must be all or development")
    num_classes = 2 if task == "binary" else 3
    development = None
    if data_scope == "development":
        if split_root is not None:
            raise ValueError("development classification requires split_root=None")
        development = evaluate_development_probabilities(
            bank, labels, fold_protocol=fold_protocol, model_name=model_name, k=k,
            seeds=seeds, num_classes=num_classes, make_classifier=make_classifier,
            predict_proba_full=predict_proba_full)
        oof = development["oof"]
    else:
        train_ids = bank["train_ids"]
        id_to_index = {pid: idx for idx, pid in enumerate(train_ids)}
        test_ids = bank["test_ids"]
        train_x = bank["train_x"]
        test_x = bank["test_x"]
        oof_lists: dict[int, list[np.ndarray]] = defaultdict(list)
        test_probs: list[np.ndarray] = []

        for fold in range(10):
            payload = create_train_val_split(
                split_csv=split_root / task / f"fold{fold}.csv",
                task=task,
                val_ratio=0.2,
                respect_split_column=True,
            )
            fold_train = [id_to_index[pid] for pid in payload["train_ids"] if pid in id_to_index]
            fold_val_ids = [pid for pid in payload["val_ids"] if pid in id_to_index]
            fold_val = [id_to_index[pid] for pid in fold_val_ids]
            y_train = np.asarray([labels[pid] for pid in payload["train_ids"] if pid in id_to_index], dtype=int)
            for seed in seeds:
                model = make_classifier(model_name, min(k, train_x.shape[1]), train_x.shape[1], seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train_x[fold_train], y_train)
                val_prob = predict_proba_full(model, train_x[fold_val], num_classes)
                for pid, probs in zip(fold_val_ids, val_prob):
                    oof_lists[pid].append(probs)
                test_probs.append(predict_proba_full(model, test_x, num_classes))

        oof = {pid: np.mean(vals, axis=0) for pid, vals in oof_lists.items()}
        test = {pid: probs for pid, probs in zip(test_ids, np.mean(test_probs, axis=0))}
    try:
        if task == "binary":
            sweep = sweep_binary_threshold(oof, labels, threshold_grid("0.25:0.75:0.01"))
            best = sweep["best"]
            threshold = float(best["threshold"])
            y_true = [labels[pid] for pid in sorted(oof)]
            y_pred = [1 if float(oof[pid][1]) >= threshold else 0 for pid in sorted(oof)]
            rule = {"threshold": threshold}
        else:
            sweep = sweep_ternary_rule(
                oof, labels, threshold_grid("0.25:0.65:0.025"),
                threshold_grid("-0.5:0.5:0.05"))
            best = sweep["best"]
            y_true = [labels[pid] for pid in sorted(oof)]
            y_pred = [apply_ternary_rule(oof[pid], best) for pid in sorted(oof)]
            rule = compact_rule(best)
        result = {
            "name": f"{bank['audio_feature']}_{bank['video_feature']}_{bank['mode']}_{model_name}_k{k}",
            "task": task, "oof": oof, "rule": rule,
            "selection": float(best["selection"]), "f1": macro_f1(y_true, y_pred),
            "kappa": cohen_kappa(y_true, y_pred),
            "dist": dict(zip(*np.unique(y_pred, return_counts=True))),
        }
        if development is None:
            result["test"] = test
        else:
            result.update(development)
        return result
    except Exception as exc:
        if development is not None:
            exc.fit_calls = int(development["fit_calls"])
            exc.completed_folds = int(development["completed_folds"])
        raise


def calibrate_regression_candidate(reg: dict[str, Any], phq: dict[int, float], shrink: float) -> dict[str, Any]:
    ids = sorted(set(reg["oof"]) & set(phq))
    y_true = np.asarray([float(phq[pid]) for pid in ids], dtype=np.float64)
    y_pred = np.asarray([float(reg["oof"][pid]) for pid in ids], dtype=np.float64)
    pred_mean = float(y_pred.mean())
    pred_std = float(y_pred.std())
    true_mean = float(y_true.mean())
    true_std = float(y_true.std())
    if pred_std <= 1e-8 or true_std <= 1e-8:
        scale = 1.0
        bias = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
        direction = 1.0 if not np.isfinite(corr) or corr >= 0 else -1.0
        scale = direction * shrink * true_std / pred_std
        bias = true_mean - scale * pred_mean

    def apply(value: float) -> float:
        return float(np.clip(scale * float(value) + bias, 0.0, 27.0))

    oof = {pid: apply(reg["oof"][pid]) for pid in reg["oof"]}
    test = {pid: apply(reg["test"][pid]) for pid in reg["test"]}
    y_cal = np.asarray([oof[pid] for pid in ids], dtype=np.float64)
    return {
        "name": f"{reg['name']}_oofcal_s{shrink:g}",
        "oof": oof,
        "test": test,
        **phq_regression_metrics(y_true, y_cal),
        "calibration": {"space": "raw", "scale": scale, "bias": bias, "shrink": shrink},
    }


def binary_rule_from_phq(oof: dict[int, float], labels: dict[int, int]) -> dict[str, Any]:
    ids = sorted(set(oof) & set(labels))
    y_true = [int(labels[pid]) for pid in ids]
    rows = []
    for threshold in np.linspace(0.0, 17.0, 69):
        y_pred = [1 if float(oof[pid]) >= threshold else 0 for pid in ids]
        f1 = macro_f1(y_true, y_pred)
        kappa = cohen_kappa(y_true, y_pred)
        rows.append(
            {
                "threshold": float(threshold),
                "f1": f1,
                "kappa": kappa,
                "selection": (f1 + kappa) / 2.0,
                "dist": dict(Counter(y_pred)),
            }
        )
    rows.sort(key=lambda row: (row["selection"], row["f1"], row["kappa"], -abs(float(row["threshold"]) - 5.0)), reverse=True)
    return rows[0]


def ternary_rule_from_phq(oof: dict[int, float], labels: dict[int, int]) -> dict[str, Any]:
    ids = sorted(set(oof) & set(labels))
    y_true = [int(labels[pid]) for pid in ids]
    rows = []
    thresholds = np.linspace(0.0, 17.0, 69)
    for low in thresholds:
        for high in thresholds:
            if high <= low:
                continue
            y_pred = []
            for pid in ids:
                pred = float(oof[pid])
                if pred < low:
                    y_pred.append(0)
                elif pred < high:
                    y_pred.append(1)
                else:
                    y_pred.append(2)
            f1 = macro_f1(y_true, y_pred)
            kappa = cohen_kappa(y_true, y_pred)
            rows.append(
                {
                    "low": float(low),
                    "high": float(high),
                    "f1": f1,
                    "kappa": kappa,
                    "selection": (f1 + kappa) / 2.0,
                    "dist": dict(Counter(y_pred)),
                }
            )
    rows.sort(key=lambda row: (row["selection"], row["f1"], row["kappa"], -abs(float(row["low"]) - 5.0), -abs(float(row["high"]) - 10.0)), reverse=True)
    return rows[0]


def evaluate_severity_candidate(
    bank: dict[str, Any],
    ternary_labels: dict[int, int],
    model_name: str,
    k: int,
    seeds: tuple[int, ...],
    *,
    data_scope: str = "all",
    fold_protocol=None,
) -> dict[str, Any]:
    if data_scope not in ("all", "development"):
        raise ValueError("data_scope must be all or development")
    development = None
    if data_scope == "development":
        train_ids = list(bank["train_ids"])
        validated_ternary = validate_classification_labels(ternary_labels, train_ids, 3)
        positive_ids = [pid for pid in train_ids if validated_ternary[pid] in (1, 2)]
        severity_labels = {pid: (1 if validated_ternary[pid] == 2 else 0)
                           for pid in train_ids}
        development = evaluate_development_probabilities(
            bank, severity_labels, fold_protocol=fold_protocol, model_name=model_name, k=k,
            seeds=seeds, num_classes=2, make_classifier=make_classifier,
            predict_proba_full=predict_proba_full, train_subset_ids=positive_ids,
            required_train_classes={0, 1}, seed_offset_by_fold=True)
        oof = development["oof"]
    else:
        train_ids = list(bank["train_ids"])
        test_ids = list(bank["test_ids"])
        train_x = bank["train_x"]
        test_x = bank["test_x"]
        id_to_index = {pid: idx for idx, pid in enumerate(train_ids)}
        positive_ids = [pid for pid in train_ids if int(ternary_labels[pid]) in (1, 2)]
        y_positive = np.asarray([1 if int(ternary_labels[pid]) == 2 else 0 for pid in positive_ids], dtype=int)
        if len(np.unique(y_positive)) < 2:
            raise ValueError("severity candidate needs both class1 and class2")

        oof_lists: dict[int, list[np.ndarray]] = defaultdict(list)
        test_probs: list[np.ndarray] = []
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=3407)
        neg_ids = [pid for pid in train_ids if int(ternary_labels[pid]) == 0]

        for fold, (train_pos_idx, val_pos_idx) in enumerate(skf.split(np.zeros(len(positive_ids)), y_positive)):
            fold_train_ids = [positive_ids[i] for i in train_pos_idx]
            fold_val_pos_ids = [positive_ids[i] for i in val_pos_idx]
            fold_train = [id_to_index[pid] for pid in fold_train_ids]
            y_train = np.asarray([1 if int(ternary_labels[pid]) == 2 else 0 for pid in fold_train_ids], dtype=int)
            predict_train_ids = fold_val_pos_ids + neg_ids
            predict_train_idx = [id_to_index[pid] for pid in predict_train_ids]

            for seed in seeds:
                model = make_classifier(model_name, min(k, train_x.shape[1]), train_x.shape[1], seed + fold)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train_x[fold_train], y_train)
                train_prob = predict_proba_full(model, train_x[predict_train_idx], 2)
                for pid, probs in zip(predict_train_ids, train_prob):
                    oof_lists[pid].append(probs)
                test_probs.append(predict_proba_full(model, test_x, 2))

        oof = {pid: np.mean(vals, axis=0) for pid, vals in oof_lists.items()}
        test = {pid: probs for pid, probs in zip(test_ids, np.mean(test_probs, axis=0))}
    try:
        ids = sorted(oof)
        labels_for_score = validated_ternary if development is not None else ternary_labels
        true_sev = [1 if labels_for_score[pid] == 2 else 0 for pid in ids]
        rows = []
        for threshold in threshold_grid("0.15:0.85:0.025"):
            pred_sev = [1 if float(oof[pid][1]) >= threshold else 0 for pid in ids]
            f1 = macro_f1(true_sev, pred_sev)
            kappa = cohen_kappa(true_sev, pred_sev)
            rows.append({"threshold": threshold, "f1": f1, "kappa": kappa,
                         "selection": (f1 + kappa) / 2.0,
                         "dist": dict(Counter(pred_sev))})
        rows.sort(key=lambda row: (row["selection"], row["f1"], row["kappa"],
                                   -abs(float(row["threshold"]) - 0.5)), reverse=True)
        best = rows[0]
        result = {
            "name": f"{bank['audio_feature']}_{bank['video_feature']}_{bank['mode']}_{model_name}_k{k}",
            "task": "severity12", "oof": oof,
            "rule": {"threshold": float(best["threshold"])},
            "selection": float(best["selection"]), "f1": float(best["f1"]),
            "kappa": float(best["kappa"]), "dist": best["dist"],
        }
        if development is None:
            result["test"] = test
        else:
            result.update(development)
        return result
    except Exception as exc:
        if development is not None:
            exc.fit_calls = int(development["fit_calls"])
            exc.completed_folds = int(development["completed_folds"])
        raise


def combine_hier_oof(
    binary: dict[str, Any],
    severity: dict[str, Any],
    ternary_labels: dict[int, int],
    *,
    selection_metric: str = "f1_kappa",
) -> dict[str, Any]:
    if selection_metric not in ("f1_kappa", "macro_f1"):
        raise ValueError("selection_metric must be f1_kappa or macro_f1")
    ids = sorted(set(binary["oof"]) & set(severity["oof"]) & set(ternary_labels))
    y_true = [int(ternary_labels[pid]) for pid in ids]
    rows = []
    b_threshold = float(binary["rule"]["threshold"])
    for s_threshold in threshold_grid("0.15:0.85:0.025"):
        y_pred = []
        for pid in ids:
            if float(binary["oof"][pid][1]) < b_threshold:
                y_pred.append(0)
            elif float(severity["oof"][pid][1]) >= s_threshold:
                y_pred.append(2)
            else:
                y_pred.append(1)
        f1 = macro_f1(y_true, y_pred)
        kappa = cohen_kappa(y_true, y_pred)
        rows.append(
            {
                "binary_threshold": b_threshold,
                "severity_threshold": s_threshold,
                "f1": f1,
                "kappa": kappa,
                "selection": (f1 + kappa) / 2.0,
                "dist": dict(Counter(y_pred)),
            }
        )
    if selection_metric == "f1_kappa":
        rows.sort(key=lambda row: (row["selection"], row["f1"], row["kappa"],
                                   -abs(float(row["severity_threshold"]) - 0.5)), reverse=True)
    else:
        rows.sort(key=lambda row: (row["f1"], row["kappa"],
                                   -abs(float(row["severity_threshold"]) - 0.5)), reverse=True)
    best = dict(rows[0])
    if selection_metric == "macro_f1":
        best["selection"] = float(best["f1"])
    best["selection_metric"] = selection_metric
    best["threshold_selection"] = "adaptive training OOF; not nested final estimate"
    return best


def phq_log_regression_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict[str, float]:
    y_true = np.log1p(np.clip(np.asarray(y_true_raw, dtype=np.float64), 0.0, None))
    y_pred = np.log1p(np.clip(np.asarray(y_pred_raw, dtype=np.float64), 0.0, None))
    metrics = phq_regression_metrics(y_true, y_pred)
    return {
        "log_ccc": float(metrics["ccc"]),
        "log_rmse": float(metrics["rmse"]),
        "log_mae": float(metrics["mae"]),
    }


def make_regressor(model_name: str, k: int, n_features: int) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    use_pca = model_name.startswith("pca")
    base_name = model_name[3:] if use_pca else model_name
    if use_pca:
        steps.append(("pca", PCA(n_components=min(k, n_features), whiten=True, random_state=3407)))
    elif k > 0 and k < n_features:
        steps.append(("select", SelectKBest(f_regression, k=k)))
    if base_name.startswith("ridge"):
        alpha = float(base_name.split("a", 1)[1])
        model: BaseEstimator = Ridge(alpha=alpha)
    elif base_name.startswith("enet"):
        payload = base_name.split("a", 1)[1]
        alpha_s, l1_s = payload.split("_l1")
        model = ElasticNet(alpha=float(alpha_s), l1_ratio=float(l1_s), max_iter=10000, random_state=3407)
    elif base_name == "bayes":
        model = BayesianRidge()
    elif base_name.startswith("huber"):
        alpha = float(base_name.split("a", 1)[1])
        model = HuberRegressor(alpha=alpha, epsilon=1.35, max_iter=1000)
    elif base_name.startswith("svrlin"):
        c = float(base_name.split("C", 1)[1])
        model = SVR(kernel="linear", C=c, epsilon=0.2)
    elif base_name.startswith("svrrbf"):
        c = float(base_name.split("C", 1)[1])
        model = SVR(kernel="rbf", C=c, epsilon=0.2, gamma="scale")
    elif base_name.startswith("krrlin"):
        alpha = float(base_name.split("a", 1)[1])
        model = KernelRidge(kernel="linear", alpha=alpha)
    elif base_name.startswith("krrrbf"):
        alpha = float(base_name.split("a", 1)[1])
        model = KernelRidge(kernel="rbf", alpha=alpha)
    elif base_name.startswith("rfleaf"):
        min_leaf = int(base_name.split("leaf", 1)[1])
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=3,
            min_samples_leaf=min_leaf,
            random_state=3407,
            n_jobs=1,
        )
    elif base_name.startswith("etleaf"):
        min_leaf = int(base_name.split("leaf", 1)[1])
        model = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=3,
            min_samples_leaf=min_leaf,
            random_state=3407,
            n_jobs=1,
        )
    elif base_name.startswith("gbr"):
        payload = base_name.split("gbr", 1)[1]
        lr_s, depth_s = payload.split("d", 1)
        model = GradientBoostingRegressor(
            learning_rate=float(lr_s),
            max_depth=int(depth_s),
            n_estimators=120,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=3407,
        )
    elif base_name.startswith("hgb"):
        payload = base_name.split("hgb", 1)[1]
        lr_s, l2_s = payload.split("l", 1)
        model = HistGradientBoostingRegressor(
            learning_rate=float(lr_s),
            l2_regularization=float(l2_s),
            max_iter=120,
            max_leaf_nodes=7,
            min_samples_leaf=5,
            early_stopping=True,
            random_state=3407,
        )
    elif base_name.startswith("knnr"):
        n_neighbors = int(base_name.split("knnr", 1)[1])
        model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    elif base_name.startswith("plsn"):
        n_components = int(base_name.split("n", 1)[1])
        model = PLSRegression(n_components=n_components)
    elif base_name.startswith("mlp"):
        payload = base_name.split("mlp", 1)[1]
        hidden_s, alpha_s = payload.split("a", 1)
        hidden = tuple(int(item) for item in hidden_s.split("_") if item)
        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=float(alpha_s),
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
            max_iter=1500,
            random_state=3407,
        )
    else:
        raise ValueError(f"Unknown regressor: {model_name}")
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate_regression_candidate(
    bank: dict[str, Any],
    phq: dict[int, float],
    split_roots: list[Path],
    model_name: str,
    k: int,
    n_folds: int,
    *,
    data_scope: str = "all",
    fold_protocol=None,
) -> dict[str, Any]:
    if data_scope not in ("all", "development"):
        raise ValueError("data_scope must be all or development")
    if data_scope == "development":
        from decimal import Decimal, InvalidOperation
        # Package import first, so the caller and this module share one DevelopmentFoldProtocol class.
        try:
            from merid.mpdd2026_protocol import iter_development_folds
        except ModuleNotFoundError as exc:
            if exc.name not in ("merid", "merid.mpdd2026_protocol"):
                raise
            from mpdd2026_protocol import iter_development_folds

        if split_roots is not None:
            raise ValueError("development regression requires split_roots=None")
        if fold_protocol is None:
            raise ValueError("development regression requires fold_protocol")
        if type(n_folds) is not int or n_folds != fold_protocol.n_folds:
            raise ValueError("n_folds must match the fixed development protocol")
        if bank.get("data_scope") != "development":
            raise ValueError("development regression requires a development bank")
        if "test_x" in bank or "test_ids" in bank:
            raise ValueError("development regression bank must not contain test data")

        train_ids = list(bank["train_ids"])
        train_x = np.asarray(bank["train_x"])
        if (train_x.ndim != 2 or train_x.shape[0] != len(train_ids) or train_x.shape[1] == 0
                or not np.isfinite(train_x).all()):
            raise ValueError("development regression requires a finite aligned feature matrix")
        folds = iter_development_folds(fold_protocol, train_ids)

        def normalized_id(value):
            try:
                number = Decimal(str(value).strip())
            except InvalidOperation as exc:
                raise ValueError("PHQ patient IDs must be finite and integral") from exc
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("PHQ patient IDs must be finite and integral")
            return str(int(number))

        targets = {}
        for patient, value in phq.items():
            normalized = normalized_id(patient)
            if normalized in targets:
                raise ValueError("PHQ patient IDs must be unique after normalization")
            target = float(value)
            if not np.isfinite(target):
                raise ValueError("PHQ targets must be finite")
            targets[normalized] = target
        if set(targets) != set(fold_protocol.patient_ids):
            raise ValueError("PHQ targets must exactly cover development protocol patients")

        oof = {}
        fit_calls, completed_folds = 0, 0
        try:
            for fold in folds:
                y_train = np.asarray([targets[patient] for patient in fold.train_ids], dtype=np.float64)
                model = make_regressor(model_name, min(k, train_x.shape[1]), train_x.shape[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit_calls += 1
                    model.fit(train_x[list(fold.train_indices)], y_train)
                val_pred = np.asarray(model.predict(train_x[list(fold.validation_indices)]),
                                      dtype=np.float64)
                if val_pred.shape == (len(fold.validation_ids), 1):
                    val_pred = val_pred[:, 0]
                if val_pred.shape != (len(fold.validation_ids),) or not np.isfinite(val_pred).all():
                    raise ValueError("development regression predictions must be finite and complete")
                val_pred = np.clip(val_pred, 0.0, 27.0)
                for patient, pred in zip(fold.validation_ids, val_pred):
                    if patient in oof:
                        raise ValueError("development regression produced duplicate OOF predictions")
                    oof[patient] = float(pred)
                completed_folds += 1
        except Exception as exc:
            exc.fit_calls = fit_calls
            exc.completed_folds = completed_folds
            raise
        if set(oof) != set(fold_protocol.patient_ids):
            raise ValueError("development regression OOF predictions must cover every patient")
        patient_ids = list(fold_protocol.patient_ids)
        y_true = np.asarray([targets[patient] for patient in patient_ids], dtype=np.float64)
        y_pred = np.asarray([oof[patient] for patient in patient_ids], dtype=np.float64)
        metrics = phq_regression_metrics(y_true, y_pred)
        return {
            "name": f"{bank['audio_feature']}_{bank['video_feature']}_{bank['mode']}_{model_name}_k{k}_fixedcv1",
            "oof": {int(patient): oof[patient] for patient in patient_ids},
            "patient_ids": [int(patient) for patient in patient_ids],
            "protocol_fingerprint": fold_protocol.protocol_fingerprint,
            "protocol_evidence_fingerprint": fold_protocol.evidence_fingerprint,
            "source_ids_fingerprint": folds[0].source_ids_fingerprint,
            "data_scope": "development",
            "fit_calls": fit_calls,
            "completed_folds": completed_folds,
            **metrics,
            **phq_log_regression_metrics(y_true, y_pred),
            "root_ccc_mean": float(metrics["ccc"]),
            "root_ccc_min": float(metrics["ccc"]),
            "root_ccc_std": 0.0,
            "n_protocols": 1,
        }
    train_ids = bank["train_ids"]
    id_to_index = {pid: idx for idx, pid in enumerate(train_ids)}
    test_ids = bank["test_ids"]
    train_x = bank["train_x"]
    test_x = bank["test_x"]
    oof_vals: dict[int, list[float]] = {}
    test_vals: list[np.ndarray] = []
    root_cccs: list[float] = []
    for split_root in split_roots:
        root_oof_vals: dict[int, list[float]] = {}
        for fold in range(n_folds):
            payload = create_train_val_split(
                split_csv=split_root / "binary" / f"fold{fold}.csv",
                task="binary",
                val_ratio=0.2,
                respect_split_column=True,
            )
            fold_train_ids = [pid for pid in payload["train_ids"] if pid in id_to_index]
            fold_val_ids = [pid for pid in payload["val_ids"] if pid in id_to_index]
            fold_train = [id_to_index[pid] for pid in fold_train_ids]
            fold_val = [id_to_index[pid] for pid in fold_val_ids]
            y_train = np.asarray([phq[pid] for pid in fold_train_ids], dtype=np.float64)
            model = make_regressor(model_name, min(k, train_x.shape[1]), train_x.shape[1])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_x[fold_train], y_train)
            val_pred = np.asarray(model.predict(train_x[fold_val]), dtype=np.float64).reshape(-1)
            val_pred = np.clip(val_pred, 0.0, 27.0)
            for pid, pred in zip(fold_val_ids, val_pred):
                oof_vals.setdefault(pid, []).append(float(pred))
                root_oof_vals.setdefault(pid, []).append(float(pred))
            test_pred = np.asarray(model.predict(test_x), dtype=np.float64).reshape(-1)
            test_vals.append(np.clip(test_pred, 0.0, 27.0))
        if root_oof_vals:
            root_oof = {pid: float(np.mean(vals)) for pid, vals in root_oof_vals.items()}
            root_ids = sorted(root_oof)
            root_metrics = phq_regression_metrics(
                np.asarray([phq[pid] for pid in root_ids], dtype=np.float64),
                np.asarray([root_oof[pid] for pid in root_ids], dtype=np.float64),
            )
            root_cccs.append(float(root_metrics["ccc"]))

    oof = {pid: float(np.mean(vals)) for pid, vals in oof_vals.items()}
    test = {pid: float(pred) for pid, pred in zip(test_ids, np.mean(test_vals, axis=0))}
    ids = sorted(oof)
    y_true = np.asarray([phq[pid] for pid in ids], dtype=np.float64)
    y_pred = np.asarray([oof[pid] for pid in ids], dtype=np.float64)
    root_ccc_mean = float(np.mean(root_cccs)) if root_cccs else 0.0
    root_ccc_min = float(np.min(root_cccs)) if root_cccs else 0.0
    root_ccc_std = float(np.std(root_cccs)) if root_cccs else 0.0
    return {
        "name": f"{bank['audio_feature']}_{bank['video_feature']}_{bank['mode']}_{model_name}_k{k}_repcv{len(split_roots)}",
        "oof": oof,
        "test": test,
        **phq_regression_metrics(y_true, y_pred),
        **phq_log_regression_metrics(y_true, y_pred),
        "root_ccc_mean": root_ccc_mean,
        "root_ccc_min": root_ccc_min,
        "root_ccc_std": root_ccc_std,
    }
