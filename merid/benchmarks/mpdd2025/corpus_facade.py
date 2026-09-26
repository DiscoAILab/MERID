"""Runs MPDD-2025 contexts through the shared development loop."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import merid.run_corpus as _corpus
from merid.run_method import propose_recipe_space as _propose_mpdd_recipes
from merid.benchmarks.mpdd2025.decoding import GroupedDecodingClassifier, STRATEGIES
from merid.benchmarks.mpdd2025.model_bridge import build_model
from merid.run_method import ORDINAL_BASES, metric_value, CLF_METRIC_NAMES, PROBA_MODELS


TASKS = {
    'MPDD-Young': ('binary', 'ternary'),
    'MPDD-Elderly': ('binary', 'ternary', 'quinary'),
}


class MPDDCorpusFacade:
    def __init__(self, context='MPDD-Young/1s', task='binary',
                 selection_metric='official_composite', decoder_strategies=('plain',)):
        parts = context.split('/')
        if len(parts) != 2 or parts[0] not in TASKS or parts[1] not in ('1s', '5s'):
            raise ValueError('invalid MPDD-2025 context')
        if task not in TASKS[parts[0]]:
            raise ValueError('task is unavailable for MPDD-2025 track')
        if selection_metric not in CLF_METRIC_NAMES:
            raise ValueError("invalid MPDD classification selection metric")
        strategies = tuple(decoder_strategies)
        if not strategies or any(value not in STRATEGIES for value in strategies):
            raise ValueError('invalid decoder strategies')
        strategies = tuple(dict.fromkeys(strategies))
        if "plain" in strategies:
            strategies = ("plain",) + tuple(s for s in strategies if s != "plain")
        self.CTX, self.TASK, self.SEL_NAME = context, task, selection_metric
        self.decoder_strategies = strategies
        self.manifest = {
            'candidate_order': 'each original config in proposal order; plain first, then requested decoders',
            'strategies': list(strategies),
            'plain_reference': 'plain' in strategies,
            'active_selection': 'strict DevelopmentSearch promotion on fixed development metric',
            'not_active': ['repeated robust ranking', 'top-K voting', 'one-SE rule'],
        }

    def _is_reg(self):
        return False

    def _dev(self, truth, prediction):
        return metric_value(self.SEL_NAME, truth, prediction)

    def _make(self, config, n_features):
        if isinstance(config, dict) and '_decoder_strategy' in config:
            if set(config) != {'_decoder_strategy', '_base_config'}:
                raise ValueError('decoder candidate requires only strategy and base config')
            strategy = config['_decoder_strategy']
            base = deepcopy(config['_base_config'])
            return GroupedDecodingClassifier(base, task=self.TASK, strategy=strategy,
                                             metric=self.SEL_NAME)
        return build_model(config, n_features, kind='classification', task=self.TASK)

    @contextmanager
    def _scope(self):
        previous = (_corpus.CTX, _corpus.TASK, _corpus.SEL_NAME)
        _corpus.CTX, _corpus.TASK, _corpus.SEL_NAME = self.CTX, self.TASK, self.SEL_NAME
        try:
            yield
        finally:
            _corpus.CTX, _corpus.TASK, _corpus.SEL_NAME = previous

    def _expand(self, configs):
        output = []
        for config in configs:
            for strategy in self.decoder_strategies:
                if strategy.startswith("metric_bias") and config.get("model") not in PROBA_MODELS:
                    continue
                output.append(config if strategy == 'plain' else {
                    '_decoder_strategy': strategy, '_base_config': deepcopy(config)})
        if self.TASK in ('ternary', 'quinary'):
            for base in ORDINAL_BASES:
                ordinal = {'model': 'ordinal_frankhall', 'base_cfg': deepcopy(base)}
                for strategy in self.decoder_strategies:
                    if strategy in ('plain', 'subject_consensus'):
                        output.append(ordinal if strategy == 'plain' else {
                            '_decoder_strategy': strategy, '_base_config': deepcopy(ordinal)})
        return output

    def propose_clf_space(self, proposer, profile, ctx, freeze_dir, traj, kind='classification'):
        if kind != 'classification' or ctx != self.CTX:
            raise ValueError('MPDD facade supports its declared classification context only')
        with self._scope():
            configs = _corpus.propose_clf_space(proposer, profile, ctx, freeze_dir, traj, kind=kind)
        return self._expand(configs)

    def propose_recipe_space(self, proposer, report, extractors, freeze_dir, traj=None, feedback=""):
        context = (("DEVELOPMENT REVISION FEEDBACK:\n" + feedback + "\n\n") if feedback else "")
        context += str(report)
        return _propose_mpdd_recipes(proposer, context, self.CTX, freeze_dir, traj=traj)

    def propose_codegen_method(self, *args, **kwargs):
        with self._scope():
            return _corpus.propose_codegen_method(*args, **kwargs)

    def _restrict_recipe_space(self, recipes, exclude_sources=''):
        return _corpus._restrict_recipe_space(recipes, exclude_sources)

    def _review_tools(self, load, recipes, report):
        with self._scope():
            tools = _corpus._review_tools(load, recipes, report)
        def scoped_callback(callback):
            def invoke(*args, **kwargs):
                with self._scope():
                    return callback(*args, **kwargs)
            return invoke
        return {role: {name: (description, scoped_callback(callback))
                       for name, (description, callback) in role_tools.items()}
                for role, role_tools in tools.items()}
