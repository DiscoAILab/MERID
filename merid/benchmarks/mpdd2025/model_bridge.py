"""Exposes the MPDD-2025 estimator factories as DevelopmentSearch builders."""
from __future__ import annotations

from copy import deepcopy

from merid.benchmarks.mpdd2025.run_all import make_clf, make_reg
from merid.run_method import FrankHallOrdinal


CLASSIFIERS = frozenset(('svc_linear', 'svc_rbf', 'logreg', 'gb', 'rf', 'et'))
REGRESSORS = frozenset(('ridge', 'svr_linear', 'svr_rbf', 'gbr', 'rfr', 'etr'))
ORDERED_TASKS = frozenset(('ternary', 'quinary', 'tri', 'pen'))
CLASSIFICATION_TASKS = frozenset(('binary', 'ternary', 'quinary', 'bin', 'tri', 'pen'))
REGRESSION_TASKS = frozenset(('phq', 'phq_score', 'regression'))
SELECTION_ONLY = frozenset(('ensemble', 'metric_decode', 'subject_consensus',
                            'semi_supervised', 'codegen', 'topk'))


def _checked(config):
    if not isinstance(config, dict):
        raise TypeError('MPDD model config must be a dictionary')
    forbidden = sorted(SELECTION_ONLY.intersection(config))
    if forbidden:
        raise ValueError('selection/decode features are outside estimator factory: ' + ', '.join(forbidden))
    return deepcopy(config)


def _validate_task(kind, task):
    if task is None:
        raise ValueError('an explicit MPDD task is required')
    if kind == 'classification' and task not in CLASSIFICATION_TASKS:
        raise ValueError('classification builder cannot serve task: ' + str(task))
    if kind == 'regression' and task not in REGRESSION_TASKS:
        raise ValueError('regression builder cannot serve task: ' + str(task))


def _validate_keys(config, allowed, label):
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise ValueError('unsupported or ignored %s config keys: %s' %
                         (label, ', '.join(unknown)))


def build_model(config, n_features, *, kind, task=None):
    config = _checked(config)
    _validate_task(kind, task)
    if type(n_features) is not int or n_features < 1:
        raise ValueError('n_features must be a positive integer')
    model = config.get('model')
    if model == 'ordinal_frankhall':
        if kind != 'classification' or task not in ORDERED_TASKS:
            raise ValueError('Frank-Hall is supported only for ordered ternary/quinary tasks')
        base = config.get('base_cfg')
        if not isinstance(base, dict) or base.get('model') != 'logreg':
            raise ValueError('Frank-Hall requires an explicit original logreg base_cfg')
        _validate_keys(config, ('model', 'base_cfg'), 'ordinal')
        base = _checked(base)
        _validate_keys(base, ('model', 'k', 'C'), 'ordinal base')
        return FrankHallOrdinal(base)
    if kind == 'classification':
        if model not in CLASSIFIERS:
            raise ValueError('unsupported original MPDD classifier: ' + str(model))
        allowed = {'model', 'k'}
        allowed.update({'C'} if model in ('svc_linear', 'svc_rbf', 'logreg') else ())
        allowed.update({'lr', 'depth'} if model == 'gb' else ())
        _validate_keys(config, allowed, model)
        return make_clf(config, n_features)
    if kind == 'regression':
        if model not in REGRESSORS:
            raise ValueError('unsupported original MPDD regressor: ' + str(model))
        allowed = {'model', 'k'}
        allowed.update({'alpha'} if model == 'ridge' else ())
        allowed.update({'C', 'eps'} if model in ('svr_linear', 'svr_rbf') else ())
        allowed.update({'lr', 'depth'} if model == 'gbr' else ())
        _validate_keys(config, allowed, model)
        return make_reg(config, n_features)
    raise ValueError('kind must be classification or regression')


def development_builder(*, kind, task=None):
    if kind not in ('classification', 'regression'):
        raise ValueError('kind must be classification or regression')
    _validate_task(kind, task)
    return lambda config, n_features: build_model(
        config, n_features, kind=kind, task=task)
