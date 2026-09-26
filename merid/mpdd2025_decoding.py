"""Grouped, training-only decoding strategies for MPDD-2025 classifiers."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

try:
    from .mpdd2025_model_bridge import build_model, CLASSIFIERS, ORDERED_TASKS
    from .run_method import (_pool_by_group, _subject_level, _tune_bias,
                             BIAS_STEPS, PROBA_MODELS)
    from .run_method import CLF_METRIC_NAMES
except ImportError:
    from mpdd2025_model_bridge import build_model, CLASSIFIERS, ORDERED_TASKS
    from run_method import (_pool_by_group, _subject_level, _tune_bias,
                            BIAS_STEPS, PROBA_MODELS)
    from run_method import CLF_METRIC_NAMES


STRATEGIES = frozenset(('plain', 'subject_consensus', 'metric_bias',
                        'metric_bias_consensus'))


def _ids(values, n, name):
    shape = np.asarray(values, dtype=object).shape if values is not None else ()
    if shape != (n,):
        raise ValueError(name + ' must be a one-dimensional scalar sequence')
    raw = list(values) if values is not None else []
    if len(raw) != n:
        raise ValueError(name + ' must contain one nonempty value per row')
    categories = set()
    result = []
    for value in raw:
        if isinstance(value, (list, tuple, dict, set, np.ndarray)):
            raise ValueError(name + ' values must be scalars')
        if value is None:
            raise ValueError(name + ' contains a missing identity')
        categories.add('text' if isinstance(value, (str, np.str_)) else 'numeric')
        text = str(value).strip()
        if not text or text.lower() in ('nan', 'none', 'null', 'inf', '-inf'):
            raise ValueError(name + ' contains a missing identity')
        try:
            number = Decimal(text)
            if not number.is_finite():
                raise ValueError(name + ' contains a nonfinite identity')
            if number == number.to_integral_value():
                text = str(int(number))
        except InvalidOperation:
            pass
        result.append(text)
    if len(categories) > 1:
        raise ValueError(name + ' mixes textual and numeric identity types')
    return np.asarray(result)


def _aligned_proba(model, X, classes):
    if not hasattr(model, 'predict_proba') or not hasattr(model, 'classes_'):
        raise ValueError('metric-bias strategy requires a probability estimator')
    raw = np.asarray(model.predict_proba(X), float)
    model_classes = np.asarray(model.classes_)
    if raw.ndim != 2 or raw.shape != (len(X), len(model_classes)) or not np.isfinite(raw).all():
        raise ValueError('estimator returned invalid probabilities')
    if (len(set(model_classes.tolist())) != len(model_classes)
            or not set(model_classes.tolist()).issubset(set(classes.tolist()))):
        raise ValueError('estimator probability classes are duplicated or unknown')
    if (np.any(raw < 0) or np.any(raw > 1)
            or not np.allclose(raw.sum(axis=1), 1.0, rtol=1e-7, atol=1e-9)):
        raise ValueError('estimator returned invalid probabilities')
    aligned = np.zeros((len(X), len(classes)), float)
    positions = {value: index for index, value in enumerate(model_classes.tolist())}
    for index, value in enumerate(classes.tolist()):
        if value in positions:
            aligned[:, index] = raw[:, positions[value]]
    return aligned


class GroupedDecodingClassifier:
    def __init__(self, config, *, task, strategy, n_splits=3, seed=0,
                 metric='official_composite'):
        if strategy not in STRATEGIES:
            raise ValueError('unsupported MPDD decoding strategy')
        if task not in ('binary', 'bin', *ORDERED_TASKS):
            raise ValueError('decoding supports MPDD classification tasks only')
        if type(n_splits) is not int or n_splits < 2:
            raise ValueError('n_splits must be an integer >= 2')
        if strategy.startswith('metric_bias') and config.get('model') not in PROBA_MODELS:
            raise ValueError('metric-bias supports only original probability classifier families')
        if metric not in CLF_METRIC_NAMES:
            raise ValueError('unsupported MPDD classification metric: ' + str(metric))
        if config.get('model') not in CLASSIFIERS and config.get('model') != 'ordinal_frankhall':
            raise ValueError('unsupported original classifier config')
        self.config, self.task, self.strategy = dict(config), task, strategy
        self.n_splits, self.seed, self.metric = n_splits, int(seed), metric

    def fit_with_groups(self, X, y, groups):
        self.fit_calls_, self.failed_fit_calls_, self.base_pipeline_fit_calls_ = 0, 0, 0
        X, raw_y = np.asarray(X, float), np.asarray(y)
        if (raw_y.shape != (len(X),) or not np.issubdtype(raw_y.dtype, np.number)
                or not np.isfinite(raw_y.astype(float)).all()
                or not np.array_equal(raw_y.astype(float), raw_y.astype(int))):
            raise ValueError('classification labels must be finite integers and match rows')
        y = raw_y.astype(int)
        groups = _ids(groups, len(X), 'training groups')
        self.classes_ = np.unique(y)
        if X.ndim != 2 or len(self.classes_) < 2 or not np.isfinite(X).all():
            raise ValueError('training data must be finite with at least two classes')
        self.consensus_ = ('consensus' in self.strategy and _subject_level(y, groups))
        self.bias_ = np.zeros(len(self.classes_))
        self.fold_ids_ = []
        if self.strategy.startswith('metric_bias'):
            splitter = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed)
            oof = np.full((len(y), len(self.classes_)), np.nan)
            coverage = np.zeros(len(y), int)
            for fold, (train, validation) in enumerate(splitter.split(X, y, groups)):
                if set(groups[train]) & set(groups[validation]):
                    raise ValueError('inner grouped fold leakage')
                if len(np.unique(y[train])) < 2:
                    raise ValueError('inner training fold is degenerate')
                model = build_model(self.config, X.shape[1], kind='classification', task=self.task)
                try:
                    model.fit(X[train], y[train])
                except Exception:
                    self.fit_calls_ += max(1, int(getattr(model, 'fit_calls_', 0)))
                    self.failed_fit_calls_ += 1
                    raise
                self.fit_calls_ += max(1, int(getattr(model, 'fit_calls_', 0)))
                self.base_pipeline_fit_calls_ += 1
                oof[validation] = _aligned_proba(model, X[validation], self.classes_)
                coverage[validation] += 1
                self.fold_ids_.append({'fold': fold, 'train_groups': sorted(set(groups[train])),
                                       'validation_groups': sorted(set(groups[validation]))})
            if not np.all(coverage == 1) or not np.isfinite(oof).all():
                raise ValueError('inner OOF probabilities require exactly-once full coverage')
            self.bias_ = _tune_bias(np.log(np.clip(oof, 1e-6, None)), y,
                                    self.classes_, metric=self.metric)
        self.model_ = build_model(self.config, X.shape[1], kind='classification', task=self.task)
        try:
            self.model_.fit(X, y)
        except Exception:
            self.fit_calls_ += max(1, int(getattr(self.model_, 'fit_calls_', 0)))
            self.failed_fit_calls_ += 1
            raise
        self.fit_calls_ += max(1, int(getattr(self.model_, 'fit_calls_', 0)))
        self.base_pipeline_fit_calls_ += (sum(kind == 'model' for kind, _ in self.model_.models_)
                                          if hasattr(self.model_, 'models_') else 1)
        self.provenance_ = {
            'strategy': self.strategy, 'task': self.task, 'seed': self.seed,
            'n_splits': self.n_splits, 'metric': self.metric,
            'bias_steps': list(BIAS_STEPS), 'bias': self.bias_.tolist(),
            'consensus_enabled': bool(self.consensus_), 'fit_calls': self.fit_calls_,
            'failed_fit_calls': self.failed_fit_calls_,
            'base_pipeline_fit_calls': self.base_pipeline_fit_calls_,
            'folds': self.fold_ids_, 'config': self.config,
        }
        return self

    def fit(self, X, y, groups=None):
        if groups is None:
            raise ValueError('grouped decoding requires training groups')
        return self.fit_with_groups(X, y, groups)

    def predict_with_groups(self, X, groups):
        X = np.asarray(X, float)
        if self.strategy.startswith('metric_bias'):
            probabilities = _aligned_proba(self.model_, X, self.classes_)
            prediction = self.classes_[(np.log(np.clip(probabilities, 1e-6, None))
                                        + self.bias_).argmax(axis=1)]
        else:
            prediction = np.asarray(self.model_.predict(X), int)
        if self.consensus_:
            prediction = _pool_by_group(prediction, _ids(groups, len(X), 'query groups'))
        return np.asarray(prediction, int)

    def predict(self, X, groups=None):
        if self.consensus_ and groups is None:
            raise ValueError('consensus prediction requires query groups')
        return self.predict_with_groups(X, groups)
