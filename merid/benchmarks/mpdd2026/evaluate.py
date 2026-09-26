"""MPDD-2026 evaluation harness: development-fold scores of the binary, ternary and PHQ-9 heads."""
from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import numpy as np

TRACK_TO_SPLIT = {"Track1": "Elder", "Track2": "Young"}
# Fold-split directory of each track (placeholders).
DEFAULT_SPLIT_ROOT = "<TRACK2_SPLIT_ROOT>"
DEFAULT_SPLIT_ROOTS = {
    "Track1": "<TRACK1_SPLIT_ROOT>",
    "Track2": DEFAULT_SPLIT_ROOT,
}


class Harness:
    def __init__(self, track="Track2", split_root=None, n_folds=10, *,
                 data_scope="all", train_root=None, personality_npy=None,
                 fold_task=None):
        if track not in TRACK_TO_SPLIT:
            raise ValueError('unknown track %r; expected one of %s' %
                             (track, ', '.join(sorted(TRACK_TO_SPLIT))))
        if data_scope not in ("all", "development"):
            raise ValueError("data_scope must be all or development")
        self.track = track
        self.n_folds = n_folds
        self.data_scope = data_scope
        self.fold_protocol = None
        if data_scope == "development":
            if any(value is None for value in (train_root, personality_npy, split_root, fold_task)):
                raise ValueError("development Harness requires explicit train_root, personality_npy, split_root, and fold_task")
            from merid.benchmarks.mpdd2026.data import load_task_maps
            root = Path(train_root).resolve()
            personality = Path(personality_npy).resolve()
            if root not in personality.parents:
                raise ValueError("development personality_npy must be inside train_root")
            train_csv = root / "split_labels_train.csv"
            if not root.is_dir() or not train_csv.is_file() or not personality.is_file():
                raise ValueError("development train_root must contain labels and personality_npy")
            self.data_root = str(root)
            self.personality_npy = str(personality)
            self.binary_maps = load_task_maps(train_csv, "binary", data_scope="development")
            self.ternary_maps = load_task_maps(train_csv, "ternary", data_scope="development")
            self.labels = {
                "binary": {int(p): int(v) for p, v in self.binary_maps["train_map"].items()},
                "ternary": {int(p): int(v) for p, v in self.ternary_maps["train_map"].items()},
            }
            self.phq = {int(p): float(v) for p, v in self.binary_maps["train_phq_map"].items()}
            cohorts = (set(self.labels["binary"]), set(self.labels["ternary"]), set(self.phq))
            if not cohorts[0] or any(cohort != cohorts[0] for cohort in cohorts[1:]):
                raise ValueError("development binary, ternary, and PHQ maps must have exact matching coverage")
            if not set(self.labels["binary"].values()) <= {0, 1}:
                raise ValueError("development binary labels must be 0 or 1")
            if not set(self.labels["ternary"].values()) <= {0, 1, 2}:
                raise ValueError("development ternary labels must be 0, 1, or 2")
            if not all(math.isfinite(value) for value in self.phq.values()):
                raise ValueError("development PHQ targets must be finite")
            from merid.benchmarks.mpdd2026.protocol import load_development_fold_protocol
            resolved_splits = Path(split_root).resolve()
            self.fold_protocol = load_development_fold_protocol(
                resolved_splits, fold_task, cohorts[0], n_folds)
            self.split_roots = [resolved_splits]
            self._bank_cache = {}
            self._abank_cache = {}
            return

        if any(value is not None for value in (train_root, personality_npy, fold_task)):
            raise ValueError("explicit development roots require data_scope='development'")
        if split_root is None:
            split_root = DEFAULT_SPLIT_ROOTS[track]
        from merid.benchmarks.mpdd2026.data import load_task_maps
        from merid.benchmarks.mpdd2026.data import resolve_project_path
        split = TRACK_TO_SPLIT[track]
        self.data_root = f"MPDD-AVG2026/MPDD-AVG2026-trainval/{split}"
        self.personality_npy = f"MPDD-AVG2026/MPDD-AVG2026-trainval/{split}/descriptions_embeddings_with_ids.npy"
        train_csv = f"MPDD-AVG2026/MPDD-AVG2026-trainval/{split}/split_labels_train.csv"

        self.binary_maps = load_task_maps(train_csv, "binary")
        self.ternary_maps = load_task_maps(train_csv, "ternary")
        self.labels = {
            "binary": {int(p): int(v) for p, v in self.binary_maps["train_map"].items()},
            "ternary": {int(p): int(v) for p, v in self.ternary_maps["train_map"].items()},
        }
        self.phq = {int(p): float(v) for p, v in self.binary_maps["train_phq_map"].items()}
        self.split_roots = [resolve_project_path(split_root)]
        self._bank_cache: dict = {}
        self._abank_cache: dict = {}

    def _bank(self, audio, video, mode):
        data_scope = getattr(self, "data_scope", "all")
        key = (data_scope, audio, video, mode)
        if key not in self._bank_cache:
            from merid.benchmarks.mpdd2026.data import build_feature_bank
            args = (self.track, audio, video, mode, self.binary_maps, self.data_root,
                    self.personality_npy)
            self._bank_cache[key] = (build_feature_bank(*args, data_scope="development")
                                     if data_scope == "development" else build_feature_bank(*args))
            if data_scope == "development":
                bank = self._bank_cache[key]
                if "test_x" in bank or "test_ids" in bank:
                    raise ValueError("development bank must not contain test data")
                from merid.benchmarks.mpdd2026.protocol import iter_development_folds
                iter_development_folds(self.fold_protocol, bank["train_ids"])
        return self._bank_cache[key]

    # PHQ-9 cut that defines the binary label on this corpus.
    LABEL_CUT = 5.0

    def _binary_rule(self, config, reg):
        if (config or {}).get("binary_rule", "oof_tuned") != "label_cut":
            from merid.benchmarks.mpdd2026.heads import binary_rule_from_phq
            return binary_rule_from_phq(reg["oof"], self.labels["binary"])
        import collections
        from sklearn.metrics import cohen_kappa_score, f1_score
        ids = sorted(set(reg["oof"]) & set(self.labels["binary"]))
        yt = [int(self.labels["binary"][i]) for i in ids]
        yp = [1 if float(reg["oof"][i]) >= self.LABEL_CUT else 0 for i in ids]
        f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        kap = float(cohen_kappa_score(yt, yp)) if len(set(yt)) > 1 else 0.0
        return {"threshold": float(self.LABEL_CUT), "f1": f1, "kappa": kap,
                "selection": (f1 + kap) / 2.0, "dist": dict(collections.Counter(yp)),
                "source": "label definition (provenance-verified cut)"}

    def evaluate(self, config: dict) -> dict:
        from merid.benchmarks.mpdd2026.heads import calibrate_regression_candidate
        from merid.benchmarks.mpdd2026.heads import binary_rule_from_phq, ternary_rule_from_phq
        from merid.benchmarks.mpdd2026.heads import evaluate_regression_candidate

        bank = self._bank(config["audio"], config["video"], config["mode"])
        if self.data_scope == "development":
            reg = evaluate_regression_candidate(
                bank, self.phq, None, config["model"], int(config["k"]), self.n_folds,
                data_scope="development", fold_protocol=self.fold_protocol)
        else:
            reg = evaluate_regression_candidate(bank, self.phq, self.split_roots,
                                                config["model"], int(config["k"]), self.n_folds)
        try:
            return self._score_phq_component(config, reg)
        except Exception as exc:
            if self.data_scope == 'development':
                exc.fit_calls = reg.get('fit_calls', 0)
                exc.completed_folds = reg.get('completed_folds', 0)
            raise

    def _score_phq_component(self, config, reg):
        from merid.benchmarks.mpdd2026.heads import calibrate_regression_candidate
        from merid.benchmarks.mpdd2026.heads import binary_rule_from_phq, ternary_rule_from_phq

        if config.get("shrink"):
            if self.data_scope == "development":
                evidence = {key: reg[key] for key in (
                    "protocol_fingerprint", "protocol_evidence_fingerprint",
                    "source_ids_fingerprint", "patient_ids", "data_scope",
                    "fit_calls", "completed_folds", "n_protocols") if key in reg}
                reg = calibrate_regression_candidate(
                    {**reg, "test": {}}, self.phq, shrink=float(config["shrink"]))
                reg.pop("test", None)
                reg.update(evidence)
            else:
                reg = calibrate_regression_candidate(reg, self.phq, shrink=float(config["shrink"]))
        log1p_phq_metrics = None
        if self.data_scope == "development":
            from merid.benchmarks.mpdd2026.metrics import METRIC_CONTRACT, raw_phq_metrics
            expected_ids = set(self.fold_protocol.patient_ids)
            oof_by_id = {str(key): float(value) for key, value in reg["oof"].items()}
            target_by_id = {str(key): float(value) for key, value in self.phq.items()}
            if set(oof_by_id) != expected_ids or set(target_by_id) != expected_ids:
                raise ValueError("development PHQ values must exactly cover the fold protocol")
            ordered_ids = list(self.fold_protocol.patient_ids)
            raw = raw_phq_metrics(
                [target_by_id[patient] for patient in ordered_ids],
                [oof_by_id[patient] for patient in ordered_ids])
            log1p_phq_metrics = {
                "contract": "phq_regression_metrics_log1p",
                **{key: float(reg[key]) for key in ("ccc", "rmse", "mae")},
            }
            reg.update(raw)
            reg["phq_metric_contract"] = METRIC_CONTRACT
        b = self._binary_rule(config, reg)
        t = ternary_rule_from_phq(reg["oof"], self.labels["ternary"])
        binF1, terF1, ccc = float(b["f1"]), float(t["f1"]), float(reg["ccc"])
        result = {
            "track": (binF1 + terF1 + ccc) / 3.0,
            "binF1": binF1, "terF1": terF1, "ccc": ccc,
            "name": reg.get("name", ""), "config": config,
        }
        if self.data_scope == "development":
            def serial(value):
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, dict):
                    return {str(key): serial(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [serial(item) for item in value]
                return value
            if "patient_ids" not in reg or "fit_calls" not in reg:
                raise ValueError("development regression evidence is missing patient_ids or fit_calls")
            patient_ids = [str(value) for value in reg["patient_ids"]]
            expected_ids = set(self.fold_protocol.patient_ids)
            if (len(patient_ids) != len(set(patient_ids))
                    or set(patient_ids) != expected_ids):
                raise ValueError("development regression patient_ids do not exactly cover the fold protocol")
            oof_by_id = {str(key): value for key, value in reg["oof"].items()}
            if set(oof_by_id) != expected_ids:
                raise ValueError("development PHQ OOF predictions do not exactly cover the fold protocol")
            phq_oof = np.asarray([oof_by_id[patient] for patient in patient_ids], dtype=float)
            if phq_oof.ndim != 1 or not np.all(np.isfinite(phq_oof)):
                raise ValueError("development PHQ OOF predictions must be finite scalars")
            fit_calls = reg["fit_calls"]
            if isinstance(fit_calls, bool) or not isinstance(fit_calls, (int, np.integer)) or fit_calls < 1:
                raise ValueError("development regression fit_calls must be an integer >= 1")
            result.update(
                evaluation_protocol="adaptive_oof_threshold_calibration",
                nested_final_estimate=False,
                protocol_fingerprint=self.fold_protocol.protocol_fingerprint,
                protocol_evidence_fingerprint=self.fold_protocol.evidence_fingerprint,
                patient_ids=patient_ids,
                phq_oof=serial(phq_oof),
                fit_calls=int(fit_calls),
                completed_folds=reg.get('completed_folds'),
                phq_metric_contract=reg["phq_metric_contract"],
                log1p_phq_metrics=serial(log1p_phq_metrics),
                rmse=float(reg["rmse"]), mae=float(reg["mae"]), r2=float(reg["r2"]),
                calibration=serial(reg.get("calibration", {
                    "shrink": config.get("shrink"), "source": "outer-train OOF"})),
                binary_rule=serial(b), ternary_rule=serial(t),
                data_scope="development",
            )
        return result

    def _abank(self, audio, video, mode="alignment"):
        data_scope = getattr(self, "data_scope", "all")
        key = (data_scope, audio, video, mode)
        if key not in self._abank_cache:
            from merid.benchmarks.mpdd2026.data import build_alignment_bank
            args = (self.track, audio, video, mode, self.binary_maps, self.data_root,
                    self.personality_npy)
            self._abank_cache[key] = (build_alignment_bank(*args, data_scope="development")
                                      if data_scope == "development" else build_alignment_bank(*args))
            if data_scope == "development":
                bank = self._abank_cache[key]
                if "test_x" in bank or "test_ids" in bank:
                    raise ValueError("development alignment bank must not contain test data")
                from merid.benchmarks.mpdd2026.protocol import iter_development_folds
                iter_development_folds(self.fold_protocol, bank["train_ids"])
        return self._abank_cache[key]

    def _sev_bank(self, cfg):
        align_mode = cfg.get("align_mode", "alignment")
        align = self._abank(cfg["audio"], cfg["video"], align_mode)
        fusion = cfg.get("sev_fusion")
        if not fusion:
            return align
        ba, bv = fusion.split(":")
        bm = cfg.get("sev_base_mode", "stats_q_pz")
        data_scope = getattr(self, "data_scope", "all")
        key = (data_scope, "sevfuse", cfg["audio"], cfg["video"], align_mode, fusion, bm)
        if key not in self._abank_cache:
            from merid.benchmarks.mpdd2026.data import concatenate_banks
            banks = [align, self._bank(ba, bv, bm)]
            self._abank_cache[key] = (
                concatenate_banks("align_base", banks, data_scope="development")
                if data_scope == "development" else concatenate_banks("align_base", banks))
        return self._abank_cache[key]

    def evaluate_ternary(self, config: dict) -> dict:
        if self.data_scope == "development":
            return self._evaluate_hierarchical_component(config)
        from merid.benchmarks.mpdd2026.heads import evaluate_classifier_candidate
        from merid.benchmarks.mpdd2026.heads import combine_hier_oof, evaluate_severity_candidate

        gate_bank = self._abank(config["audio"], config["video"], config.get("align_mode", "alignment"))
        sev_bank = self._sev_bank(config)
        sr = self.split_roots[0]
        seeds = (1, 42, 3407)
        k = int(config["k"])
        gate = evaluate_classifier_candidate(gate_bank, self.labels["binary"], sr, "binary",
                                             config["gate_model"], k, seeds)
        sev = evaluate_severity_candidate(sev_bank, self.labels["ternary"], config["sev_model"], k, seeds)
        rule = combine_hier_oof(gate, sev, self.labels["ternary"])
        return {"terF1": float(rule["f1"]), "terKappa": float(rule["kappa"]), "rule": rule, "config": config}

    def _evaluate_hierarchical_component(self, config):
        from merid.benchmarks.mpdd2026.heads import evaluate_classifier_candidate
        from merid.benchmarks.mpdd2026.heads import combine_hier_oof, evaluate_severity_candidate
        from merid.benchmarks.mpdd2026.metrics import METRIC_CONTRACT
        if self.data_scope != 'development' or self.fold_protocol is None:
            raise ValueError('hierarchical component requires explicit development protocol')
        gate_bank = self._abank(config['audio'], config['video'], config.get('align_mode', 'alignment'))
        sev_bank = self._sev_bank(config)
        seeds, k = (1, 42, 3407), int(config['k'])
        patients = list(self.fold_protocol.patient_ids)
        expected = set(patients)

        def checked(value):
            if ('test' in value or value.get('data_scope') != 'development'
                    or value.get('protocol_fingerprint') != self.fold_protocol.protocol_fingerprint
                    or value.get('protocol_evidence_fingerprint') != self.fold_protocol.evidence_fingerprint
                    or value.get('completed_folds') != self.n_folds
                    or value.get('fit_calls') != self.n_folds * len(seeds)):
                raise ValueError('hierarchical component protocol, coverage or fit cost mismatch')
            ids = [str(p) for p in value['patient_ids']]
            probabilities = {str(p): np.asarray(v, float) for p, v in value['oof'].items()}
            if len(ids) != len(set(ids)) or set(ids) != expected or set(probabilities) != expected:
                raise ValueError('hierarchical OOF must exactly cover all protocol patients')
            ordered = np.asarray([probabilities[p] for p in patients])
            if (ordered.shape != (len(patients), 2) or not np.isfinite(ordered).all()
                    or np.any((ordered < 0) | (ordered > 1))
                    or not np.allclose(ordered.sum(axis=1), 1., atol=1e-8)):
                raise ValueError('hierarchical probabilities must be finite normalized binary rows')
            return ordered

        fit_calls = 0
        try:
            gate = evaluate_classifier_candidate(gate_bank, self.labels['binary'], None, 'binary',
                config['gate_model'], k, seeds, data_scope='development', fold_protocol=self.fold_protocol)
            fit_calls += int(gate['fit_calls'])
            gate_oof = checked(gate)
            sev = evaluate_severity_candidate(sev_bank, self.labels['ternary'], config['sev_model'], k,
                seeds, data_scope='development', fold_protocol=self.fold_protocol)
            fit_calls += int(sev['fit_calls'])
            sev_oof = checked(sev)
            rule = combine_hier_oof(gate, sev, self.labels['ternary'], selection_metric='macro_f1')
        except Exception as exc:
            exc.fit_calls = fit_calls + int(getattr(exc, 'fit_calls', 0))
            raise
        try:
            ter_f1, kappa = float(rule['f1']), float(rule['kappa'])
            if not math.isfinite(ter_f1) or not 0 <= ter_f1 <= 1 or not math.isfinite(kappa):
                raise ValueError('hierarchical rule returned invalid metrics')
            serial_rule = {key: (value.item() if isinstance(value, np.generic) else value)
                           for key, value in rule.items()}
            return dict(terF1=ter_f1, terKappa=kappa, rule=serial_rule, config=config,
                        head_family='hierarchical_ternary',
                        evaluation_protocol='shared_fixed_outer_folds_hierarchical_ternary',
                        nested_final_estimate=False, data_scope='development',
                        protocol_fingerprint=self.fold_protocol.protocol_fingerprint,
                        protocol_evidence_fingerprint=self.fold_protocol.evidence_fingerprint,
                        phq_metric_contract=METRIC_CONTRACT,
                        patient_ids=patients, gate_oof=gate_oof.tolist(), severity_oof=sev_oof.tolist(),
                        gate_rule=gate['rule'], severity_rule=sev['rule'], seeds=list(seeds),
                        gate_source_ids_fingerprint=gate['source_ids_fingerprint'],
                        severity_source_ids_fingerprint=sev['source_ids_fingerprint'],
                        gate_fit_calls=int(gate['fit_calls']), severity_fit_calls=int(sev['fit_calls']),
                        fit_calls=fit_calls, completed_folds=self.n_folds,
                        threshold_selection='macro_f1 severity threshold on training OOF; F1+kappa gate threshold; adaptive, not nested final',
                        threshold_selection_metric='macro_f1')
        except Exception as exc:
            exc.fit_calls = fit_calls
            exc.completed_folds = self.n_folds
            raise

    def evaluate_estimator(self, build, config: dict, task: str = "binary") -> float:
        if self.data_scope == "development":
            raise RuntimeError("development generic estimator evaluation is not wired")
        import numpy as np
        from sklearn.metrics import f1_score
        from merid.benchmarks.mpdd2026.data import create_train_val_split

        bank = self._bank(config["audio"], config["video"], config["mode"])
        X = np.asarray(bank["train_x"], dtype=np.float64)
        ids = list(bank["train_ids"])
        id2i = {p: i for i, p in enumerate(ids)}
        y = np.array([self.labels[task][p] for p in ids])
        fold_task = task if task in ("binary", "ternary") else "binary"
        oof = np.full(len(ids), -1)
        for fold in range(self.n_folds):
            payload = create_train_val_split(split_csv=self.split_roots[0] / fold_task / f"fold{fold}.csv",
                                             task="binary", val_ratio=0.2, respect_split_column=True)
            tr = [id2i[p] for p in payload["train_ids"] if p in id2i]
            va = [id2i[p] for p in payload["val_ids"] if p in id2i]
            if not tr or not va:
                continue
            est = build()
            est.fit(X[tr], y[tr])
            for j, p in zip(va, est.predict(X[va])):
                oof[j] = int(p)
        m = oof >= 0
        return float(f1_score(y[m], oof[m], average="macro", zero_division=0))

    def predict_phq_binary(self, phq_cfg: dict) -> dict:
        if self.data_scope == "development":
            raise RuntimeError("development prediction is not wired")
        from merid.benchmarks.mpdd2026.heads import calibrate_regression_candidate
        from merid.benchmarks.mpdd2026.heads import binary_rule_from_phq
        from merid.benchmarks.mpdd2026.heads import evaluate_regression_candidate

        bank = self._bank(phq_cfg["audio"], phq_cfg["video"], phq_cfg["mode"])
        reg = evaluate_regression_candidate(bank, self.phq, self.split_roots, phq_cfg["model"],
                                            int(phq_cfg["k"]), self.n_folds)
        if phq_cfg.get("shrink"):
            reg = calibrate_regression_candidate(reg, self.phq, shrink=float(phq_cfg["shrink"]))
        b_thr = float(self._binary_rule(phq_cfg, reg)["threshold"])
        return {pid: {"binary": int(float(reg["test"][pid]) >= b_thr), "phq": float(reg["test"][pid])}
                for pid in reg["test"]}

    def predict_ternary(self, ternary_cfg: dict) -> dict:
        if self.data_scope == "development":
            raise RuntimeError("development prediction is not wired")
        from merid.benchmarks.mpdd2026.heads import evaluate_classifier_candidate
        from merid.benchmarks.mpdd2026.heads import combine_hier_oof, evaluate_severity_candidate

        gate_bank = self._abank(ternary_cfg["audio"], ternary_cfg["video"], ternary_cfg.get("align_mode", "alignment"))
        sev_bank = self._sev_bank(ternary_cfg)
        sr = self.split_roots[0]; seeds = (1, 42, 3407); k = int(ternary_cfg["k"])
        gate = evaluate_classifier_candidate(gate_bank, self.labels["binary"], sr, "binary", ternary_cfg["gate_model"], k, seeds)
        sev = evaluate_severity_candidate(sev_bank, self.labels["ternary"], ternary_cfg["sev_model"], k, seeds)
        t = combine_hier_oof(gate, sev, self.labels["ternary"])
        gth, sth = float(t["binary_threshold"]), float(t["severity_threshold"])
        return {pid: (0 if float(gate["test"][pid][1]) < gth else (2 if float(sev["test"][pid][1]) >= sth else 1))
                for pid in gate["test"]}

    def predict_submission(self, phq_cfg: dict, ternary_cfg: dict) -> dict:
        if self.data_scope == "development":
            raise RuntimeError("development final submission is not wired")
        pb = self.predict_phq_binary(phq_cfg)
        tern = self.predict_ternary(ternary_cfg)
        return {pid: {"binary": pb[pid]["binary"], "ternary": tern.get(pid, 1), "phq": pb[pid]["phq"]}
                for pid in pb}


def recombine(phqreg: dict, ternary: dict) -> dict:
    binF1, ccc = float(phqreg["binF1"]), float(phqreg["ccc"])
    terF1 = float(ternary["terF1"])
    return {
        "track": (binF1 + terF1 + ccc) / 3.0,
        "binF1": binF1, "terF1": terF1, "ccc": ccc,
        "binary_phq_head": phqreg.get("config"), "ternary_hier_head": ternary.get("config"),
    }


def evaluate(config: dict, harness: Harness | None = None) -> dict:
    return (harness or Harness()).evaluate(config)
