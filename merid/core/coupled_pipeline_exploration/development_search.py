"""Coupled recipe x predictor search under one fixed development protocol, with uncertainty-aware incumbent retention."""
from __future__ import annotations

import hashlib
import json
import os
import time
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold


class CheckpointPersistenceError(RuntimeError):
    pass


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _identities(values, n, name):
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != n:
        raise ValueError(name + ' must have one identity per row')
    out = []
    for value in values.tolist():
        text = str(value).strip()
        if value is None or text.lower() in ('', 'nan', 'none', 'null', 'inf', '-inf'):
            raise ValueError(name + ' contains a missing identity')
        try:
            number = Decimal(text)
            if number.is_finite() and number == number.to_integral_value():
                text = str(int(number))
        except InvalidOperation:
            pass
        out.append(text)
    return np.asarray(out)


def _estimator_protocol(model):
    fit_present = hasattr(model, 'fit_with_groups')
    predict_present = hasattr(model, 'predict_with_groups')
    if not fit_present and not predict_present:
        return 'standard'
    if not (fit_present and predict_present
            and callable(model.fit_with_groups) and callable(model.predict_with_groups)):
        raise ValueError('grouped estimator protocol requires both methods to be callable')
    return 'grouped'


def _declared_fit_calls(model):
    value = getattr(model, 'fit_calls_', 1)
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError('fit_calls_ must be an integer >= 1')
    return int(value)


def _fit_model(model, X, y, groups, record=None):
    protocol = _estimator_protocol(model)
    if record is not None:
        prior = record.get('estimator_protocol')
        if prior is not None and prior != protocol:
            raise ValueError('estimator protocol changed across fits')
        record['estimator_protocol'] = protocol
        record['fit_calls'] += 1
    fit_error = None
    try:
        if protocol == 'grouped':
            model.fit_with_groups(X, y, groups)
        else:
            model.fit(X, y)
    except BaseException as exc:
        fit_error = exc
        raise
    finally:
        try:
            calls = _declared_fit_calls(model)
        except ValueError as cost_error:
            if fit_error is None:
                raise
            if record is not None:
                record['fit_cost_error'] = str(cost_error)
        else:
            if record is not None:
                record['fit_calls'] += calls - 1
    return calls


def _predict_model(model, X, groups):
    return (model.predict_with_groups(X, groups)
            if _estimator_protocol(model) == 'grouped' else model.predict(X))


class DevelopmentSearch:
    def __init__(self, load, *, task, kind, builder, scorer, seed=0, n_splits=3, promotion_se=0.0,
                 max_evaluations=256, selection_rule='incumbent'):
        if kind not in ('classification', 'regression'):
            raise ValueError('unsupported task kind')
        if n_splits < 2 or max_evaluations < 1:
            raise ValueError('n_splits >= 2 and positive max_evaluations are required')
        self.load, self.task, self.kind = load, task, kind
        self.builder, self.scorer = builder, scorer
        self.seed, self.n_splits = int(seed), int(n_splits)
        self.promotion_se = float(promotion_se)
        self.selection_rule = str(selection_rule or 'incumbent')
        if self.selection_rule not in ('incumbent', 'one_se_of_best'):
            raise ValueError("selection_rule must be 'incumbent' or 'one_se_of_best'")
        self._venue = None
        self.max_evaluations = int(max_evaluations)
        self.views, self.load_errors, self.records = {}, {}, []
        self.best, self.protocol, self.reference = None, None, None
        self.folds, self.rounds, self.evaluated = [], [], set()
        self.selected_model, self.refit_calls = None, 0

    def _block(self, raw, split):
        X = np.asarray(raw[split + '_X'], float)
        if X.ndim != 2 or min(X.shape) == 0 or not np.isfinite(X).all():
            raise ValueError(split + ' features must be nonempty finite [rows, features]')
        y = np.asarray(raw[split + '_' + self.task], float)
        if y.shape != (len(X),) or not np.isfinite(y).all():
            raise ValueError(split + ' labels must be finite and match rows')
        if self.kind == 'classification':
            if not np.array_equal(y, y.astype(int)):
                raise ValueError('classification targets must be integers')
            y = y.astype(int)
        group_key = split + '_subject'
        if group_key not in raw:
            group_key = split + '_ids'
        if group_key not in raw:
            raise ValueError(split + ' subject identities are required; row indices are insufficient')
        groups = _identities(raw[group_key], len(X), group_key)
        ids = _identities(raw[split + '_ids'], len(X), split + '_ids') if split + '_ids' in raw else groups
        if len(set(ids)) != len(ids):
            raise ValueError(split + ' needs unique row IDs when a subject has repeated rows')
        return {split + '_X': X, split + '_' + self.task: y,
                split + '_subject': groups, split + '_ids': ids}

    def view(self, recipe):
        key = fingerprint(recipe)
        if key in self.load_errors:
            raise ValueError(self.load_errors[key])
        if key in self.views:
            return self.views[key]
        try:
            raw = self.load(recipe)
            d = self._block(raw, 'train')
            has_dev = 'dev_X' in raw
            if has_dev:
                d.update(self._block(raw, 'dev'))
                if d['train_X'].shape[1] != d['dev_X'].shape[1]:
                    raise ValueError('train/dev feature dimensions differ')
                if set(d['train_subject']) & set(d['dev_subject']):
                    raise ValueError('train/dev subject overlap')
            protocol = 'official_dev_selection' if has_dev else 'grouped_oof_selection'
            identity = {k: v for k, v in d.items() if not k.endswith('_X')}
            if self.reference is None:
                folds = []
                if not has_dev:
                    g, y = d['train_subject'], d['train_' + self.task]
                    ns = min(self.n_splits, len(set(g)))
                    if ns < 2:
                        raise ValueError('at least two subjects are required')
                    splitter = (StratifiedGroupKFold(ns, shuffle=True, random_state=self.seed)
                                if self.kind == 'classification' else GroupKFold(ns))
                    folds = list(splitter.split(d['train_X'], y, g))
                    coverage = np.zeros(len(y), int)
                    for tr, va in folds:
                        if set(g[tr]) & set(g[va]):
                            raise ValueError('group leakage in development folds')
                        coverage[va] += 1
                    if not np.all(coverage == 1):
                        raise ValueError('OOF folds must cover each row exactly once')
                self.protocol, self.reference, self.folds = protocol, identity, folds
            # Every recipe returns the same subjects, row order and labels as the first valid one,
            # so all candidates are scored on identical records.
            elif protocol != self.protocol or set(identity) != set(self.reference) or any(
                    not np.array_equal(v, self.reference[k]) for k, v in identity.items()):
                raise ValueError('recipe identities, row order or labels differ from incumbent venue')
            self.views[key] = d
            return d
        except Exception as exc:
            self.load_errors[key] = type(exc).__name__ + ': ' + str(exc)
            raise

    def _margin(self, yt, d):
        # Uncertainty-aware retention: a challenger replaces the incumbent when its gain exceeds
        # promotion_se subject-bootstrap SEs.
        if self.promotion_se <= 0 or self.best is None:
            return 1e-12
        if 'se' not in self.best:
            self.best['se'] = self._bootstrap_se(np.asarray(yt), np.asarray(self.best['prediction']),
                                                  np.asarray(self._venue))
        return max(1e-12, self.promotion_se * float(self.best['se']))

    @staticmethod
    def _complexity(config):
        if config == 'codegen' or not isinstance(config, dict):
            return (3, 10 ** 6, 9)
        family = {'svc_linear': 0, 'logreg': 0, 'ridge': 0, 'lasso': 0, 'pls': 0,
                  'svc_rbf': 1, 'svr': 1, 'svr_linear': 1, 'svr_rbf': 1,
                  'gb': 2, 'gbr': 2, 'rf': 3, 'et': 3, 'rfr': 3, 'etr': 3}
        return (0, int(config.get('k', 128)), family.get(config.get('model'), 5))

    def _apply_one_se(self):
        pool = [r for r in self.records if r.get('valid') and r.get('score') is not None]
        if not pool or self.reference is None:
            return
        top = max(pool, key=lambda r: float(r['score']))
        split = 'dev' if self.protocol == 'official_dev_selection' else 'train'
        try:
            yt = np.asarray(self.reference[split + '_' + self.task])
            groups = np.asarray(self.reference[split + '_subject'])
        except (KeyError, TypeError):
            self.best = top
            return
        if 'se' not in top:
            top['se'] = self._bootstrap_se(yt, np.asarray(top['prediction']), groups)
        floor = float(top['score']) - float(top['se'])
        tied = [r for r in pool if float(r['score']) >= floor - 1e-12]
        tied.sort(key=lambda r: (self._complexity(r['config']), -float(r['score'])))
        chosen = tied[0]
        if self.best is None or chosen['id'] != self.best['id']:
            self.selected_model = None
        self.best = chosen
        self.selected_1se = dict(top_score=float(top['score']), se=float(top['se']),
                                 floor=floor, n_tied=len(tied), selected_score=float(chosen['score']))

    def _bootstrap_se(self, y, yp, groups, n_boot=1000):
        # Subjects are resampled as units, keeping each subject's records together.
        subs = np.unique(groups)
        if len(subs) < 4 or len(yp) != len(y):
            return 0.0
        idx_of = {g: np.flatnonzero(groups == g) for g in subs}
        rng = np.random.default_rng(self.seed)
        out = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(subs), len(subs))
            rows = np.concatenate([idx_of[subs[i]] for i in pick])
            if self.kind == 'classification' and len(np.unique(y[rows])) < 2:
                continue
            try:
                out.append(float(self.scorer(y[rows], yp[rows])))
            except Exception:
                continue
        return float(np.std(out)) if out else 0.0

    def invalidate_source(self, name):
        def used(recipe):
            return bool(isinstance(recipe, dict) and recipe.get(name))

        dropped = [r for r in self.records if used(r.get('recipe'))]
        if not dropped:
            return [], False
        had_incumbent = bool(self.best is not None and used(self.best.get('recipe')))
        keep = [r for r in self.records if not used(r.get('recipe'))]
        self.records = keep
        self.evaluated = {r['id'] for r in keep if r.get('id')}
        valid = [r for r in keep if r.get('valid') and r.get('score') is not None]
        self.best = max(valid, key=lambda r: float(r['score'])) if valid else None
        if self.best is not None:
            self.best.pop('se', None)
        return dropped, had_incumbent

    def evaluate(self, recipes, configs, round_no=0, on_record=None):
        incumbent = self.best['id'] if self.best else None
        start_count = len(self.records)
        from merid.utils.progress import Progress
        planned = min(self.max_evaluations - len(self.records),
                      sum(1 for c in configs for r in recipes
                          if fingerprint({'recipe': r, 'config': c}) not in self.evaluated))
        bar = Progress('search r%d' % round_no, planned)
        n_fail = 0
        for cfg in configs:
            for recipe in recipes:
                key = fingerprint({'recipe': recipe, 'config': cfg})
                if key in self.evaluated:
                    continue
                if len(self.records) >= self.max_evaluations:
                    break
                self.evaluated.add(key)
                record = dict(id=key, recipe=recipe, config=cfg, round=int(round_no),
                              valid=False, fit_calls=0)
                started = time.monotonic()
                try:
                    d = self.view(recipe)
                    X, y = d['train_X'], d['train_' + self.task]
                    if self.kind == 'classification' and len(np.unique(y)) < 2:
                        raise ValueError('training target has only one class')
                    self._venue = (yt_groups := (d['dev_subject'] if self.protocol == 'official_dev_selection'
                                                 else d['train_subject']))
                    if self.protocol == 'official_dev_selection':
                        yt = d['dev_' + self.task]
                        model = self.builder(cfg, X.shape[1])
                        _fit_model(model, X, y, d['train_subject'], record)
                        prediction = np.asarray(_predict_model(
                            model, d['dev_X'], d['dev_subject']))
                    else:
                        yt = y
                        prediction = np.full(len(y), np.nan)
                        for tr, va in self.folds:
                            if self.kind == 'classification' and len(np.unique(y[tr])) < 2:
                                raise ValueError('a training fold has only one class; cannot report partial OOF')
                            model = self.builder(cfg, X.shape[1])
                            _fit_model(model, X[tr], y[tr], d['train_subject'][tr], record)
                            fold_prediction = np.asarray(_predict_model(
                                model, X[va], d['train_subject'][va]))
                            if fold_prediction.shape != (len(va),):
                                raise ValueError('estimator must return one prediction per query row')
                            prediction[va] = fold_prediction
                    if prediction.shape != yt.shape or not np.isfinite(prediction).all():
                        raise ValueError('predictions must be finite and cover the complete venue')
                    if self.kind == 'classification':
                        if not np.isin(prediction, np.unique(y)).all():
                            raise ValueError('model returned unknown classification labels')
                        prediction = prediction.astype(int)
                    score = float(self.scorer(yt, prediction))
                    if not np.isfinite(score):
                        raise ValueError('nonfinite development metric')
                    record.update(valid=True, score=score, prediction=prediction.tolist(),
                                  n_features=int(X.shape[1]))
                    if self.selection_rule == 'one_se_of_best':
                        take = self.best is None or score > float(self.best['score']) + 1e-12
                    else:
                        take = self.best is None or score > self.best['score'] + self._margin(yt, d)
                    if take:
                        self.best = record
                        self.selected_model = model if self.protocol == 'official_dev_selection' else None
                except Exception as exc:
                    record['error'] = type(exc).__name__ + ': ' + str(exc)[:500]
                record['elapsed_seconds'] = time.monotonic() - started
                self.records.append(record)
                n_fail += 0 if record.get('valid') else 1
                bar.update(note='best=%s  fails=%d' % (
                    ('%.4f' % self.best['score']) if self.best else '-', n_fail))
                if on_record is not None:
                    on_record(record)
        bar.done()
        if self.selection_rule == 'one_se_of_best':
            self._apply_one_se()
        result = dict(round=int(round_no), incumbent=incumbent,
                      selected=self.best['id'] if self.best else None,
                      promoted=self.best is not None and self.best['id'] != incumbent,
                      new_evaluations=len(self.records) - start_count,
                      budget_exhausted=len(self.records) >= self.max_evaluations)
        self.rounds.append(result)
        return result

    def checkpoint(self, output_dir, *, phase, agent_status=None, run_status='incomplete'):
        if run_status not in ('incomplete', 'complete'):
            raise ValueError('run_status must be incomplete or complete')
        try:
            return self._write_checkpoint(output_dir, phase=phase, agent_status=agent_status,
                                          run_status=run_status)
        except Exception as exc:
            raise CheckpointPersistenceError(
                'Development checkpoint persistence failed during ' + str(phase)) from exc

    def _write_checkpoint(self, output_dir, *, phase, agent_status, run_status):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        payload = _json_safe(dict(
            format_version=1, run_status=run_status, phase=str(phase),
            agent_status=agent_status or {}, search=self.summary(),
            state=dict(
                identities=self.reference or {},
                folds=[dict(train=tr, validation=va) for tr, va in self.folds],
                evaluated=sorted(self.evaluated),
                incumbent_id=self.best['id'] if self.best else None,
                records=self.records,
            )))
        destination = output / 'development_checkpoint.json'
        temporary = output / '.development_checkpoint.json.tmp'
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, destination)
        return destination

    def summary(self):
        split = 'dev' if self.protocol == 'official_dev_selection' else 'train'
        return dict(promotion_rule=(
                        'simplest candidate within one subject-bootstrap SE of the best score '
                        '(classic 1-SE rule, order-independent)'
                        if self.selection_rule == 'one_se_of_best' else
                        ('incumbent retained unless a challenger clears %g subject-bootstrap SE of '
                         'its score' % self.promotion_se) if self.promotion_se > 0 else
                        'strict improvement > 1e-12'),
                    selection_rule=self.selection_rule,
                    one_se_selection=getattr(self, 'selected_1se', None), protocol=self.protocol,
                    task=self.task, kind=self.kind, seed=self.seed,
                    selected_id=self.best['id'] if self.best else None,
                    estimator_protocol=self.best.get('estimator_protocol') if self.best else None,
                    primary_score=self.best['score'] if self.best else None,
                    max_evaluations=self.max_evaluations, n_evaluations=len(self.records),
                    fit_calls=sum(r['fit_calls'] for r in self.records) + self.refit_calls,
                    refit_calls=self.refit_calls,
                    selected_model_scope='train fit scored on dev' if split == 'dev' else
                                         'full-train refit; OOF scores belong to fold models',
                    query_split=split, rounds=self.rounds,
                    candidates=[{k: v for k, v in r.items() if k != 'prediction'} for r in self.records],
                    limitations=['Adaptive development selection scores; final evaluation is separate.',
                                 'Adapter-internal file access is not isolated by this dictionary boundary.',
                                 'Frozen-adapter preprocessing may be fitted before inner folds.',
                                 'Metric uses adapter rows; group-disjoint folds do not imply patient pooling.'])

    def save(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if self.best is not None and self.selected_model is None:
            d = self.view(self.best['recipe'])
            model = self.builder(self.best['config'], d['train_X'].shape[1])
            refit_record = {'fit_calls': 0}
            try:
                _fit_model(model, d['train_X'], d['train_' + self.task],
                           d['train_subject'], refit_record)
            finally:
                self.refit_calls += refit_record['fit_calls']
            self.selected_model = model
        summary = self.summary()
        (output / 'development_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        (output / 'development_predictions.json').write_text(json.dumps(
            dict(identities={k: v.tolist() for k, v in (self.reference or {}).items()},
                 candidates=[dict(id=r['id'], prediction=r['prediction'])
                             for r in self.records if r['valid']])), encoding='utf-8')
        (output / 'development_folds.json').write_text(json.dumps([
            dict(train=tr.tolist(), validation=va.tolist()) for tr, va in self.folds]), encoding='utf-8')
        if self.best is not None:
            from joblib.externals import cloudpickle
            with (output / 'selected_model.joblib').open('wb') as fh:
                cloudpickle.dump(self.selected_model, fh)
        return summary
