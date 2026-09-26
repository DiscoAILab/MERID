"""Selection evidence for the review roles, measured from development data alone."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def feature_equivalence(views):
    groups = {}
    for recipe_id, view in views.items():
        digest = hashlib.sha256()
        for split in ('train', 'dev'):
            key = split + '_X'
            if key not in view:
                continue
            array = np.ascontiguousarray(view[key], dtype=np.float64)
            digest.update(json.dumps([key, array.shape]).encode())
            digest.update(memoryview(array).cast('B'))
        groups.setdefault(digest.hexdigest(), []).append(recipe_id)
    return dict(distinct_feature_views=len(groups),
                groups=[sorted(ids) for ids in groups.values() if len(ids) > 1],
                meaning='Identical validated train/dev arrays on the same identities; '
                        'does not establish an implementation error or equivalence on future data.')


def explored_surface(search, recipes, configs, remaining_evaluations):
    valid = [r for r in search.records if r.get("valid") and r.get("score") is not None]

    def family(config):
        if isinstance(config, dict):
            return str(config.get("model", "unnamed"))
        return str(config)

    scored = {}
    for record in valid:
        name = family(record.get("config"))
        entry = scored.setdefault(name, dict(n_scored=0, best_score=None))
        entry["n_scored"] += 1
        if entry["best_score"] is None or record["score"] > entry["best_score"]:
            entry["best_score"] = float(record["score"])
    offered = []
    for config in configs:
        name = family(config)
        if name not in scored and name not in offered:
            offered.append(name)

    sources = {}
    for record in valid:
        for key, value in (record.get("recipe") or {}).items():
            if key == "agg" or not value:
                continue
            entry = sources.setdefault(key, dict(n_scored=0, best_score=None))
            entry["n_scored"] += 1
            if entry["best_score"] is None or record["score"] > entry["best_score"]:
                entry["best_score"] = float(record["score"])
    in_winner = sorted(k for k, v in (search.best or {}).get("recipe", {}).items()
                       if k != "agg" and v) if search.best else []

    per_round = {}
    for record in valid:
        key = int(record.get("round", 0))
        per_round[key] = max(per_round.get(key, record["score"]), float(record["score"]))
    ladder = sorted((float(r["score"]) for r in valid), reverse=True)[:5]

    return dict(
        n_evaluations=len(search.records), n_valid=len(valid),
        remaining_evaluations=int(remaining_evaluations),
        pairs_in_space=len(recipes) * len(configs),
        pairs_unscored=max(0, len(recipes) * len(configs) - len(search.records)),
        model_families_scored=scored,
        model_families_offered_but_never_scored=offered,
        sources_scored=sources, sources_in_incumbent=in_winner,
        best_score_by_round=per_round, top5_scores=ladder,
        incumbent_score=float(search.best["score"]) if search.best else None,
        scope="Coverage of the development search on the selection venue only. A family or source "
              "listed as unscored was never fitted in this run; it is an unexplored direction, not "
              "a defect. Nothing here is a measurement on held-out data.")


def _constant(y, kind):
    if not len(y):
        raise ValueError('reference has an empty training fold')
    if kind == 'regression':
        return float(np.mean(y))
    labels, counts = np.unique(y, return_counts=True)
    return labels[np.argmax(counts)]


def selection_diagnostics(search):
    if search.best is None or search.reference is None:
        raise ValueError('selection diagnostics require a validated incumbent')
    train_y = np.asarray(search.reference['train_' + search.task])
    if search.protocol == 'official_dev_selection':
        truth = np.asarray(search.reference['dev_' + search.task])
        baseline = np.full(len(truth), _constant(train_y, search.kind))
        reference_scope = 'constant fitted on training labels, evaluated on official dev'
    elif search.protocol == 'grouped_oof_selection':
        truth = train_y
        baseline = np.full(len(truth), np.nan)
        coverage = np.zeros(len(truth), dtype=int)
        for train, validation in search.folds:
            baseline[validation] = _constant(train_y[train], search.kind)
            coverage[validation] += 1
        if not np.all(coverage == 1):
            raise ValueError('reference OOF coverage must equal one for every row')
        reference_scope = 'constant fitted separately within each existing training fold'
    else:
        raise ValueError('unknown selection protocol')
    prediction = np.asarray(search.best['prediction'])
    if prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError('selected predictions do not cover the complete venue')
    reference = dict(name='training majority class' if search.kind == 'classification'
                     else 'training mean', scope=reference_scope)
    try:
        score = float(search.scorer(truth, baseline))
        if np.isfinite(score):
            reference['primary_score'] = score
        else:
            reference.update(primary_score=None, unavailable_reason='reference metric is nonfinite')
    except Exception as exc:
        reference.update(primary_score=None,
                         unavailable_reason='reference metric failed: ' + type(exc).__name__)
    report = dict(
        selected_id=search.best['id'], primary_score=float(search.best['score']),
        n_query=int(len(truth)), protocol=search.protocol,
        reference=reference,
        feature_equivalence=feature_equivalence(search.views),
        uncertainty='No confidence interval, standard error or significance test computed.',
        scope='Adaptive development selection evidence; low or high scores do not establish '
              'absence of leakage, clinical validity, or performance on final evaluation.')
    if search.kind == 'classification':
        labels = np.unique(np.concatenate([train_y, truth, prediction])).astype(int)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, prediction, labels=labels, zero_division=0)
        report.update(
            accuracy=float(accuracy_score(truth, prediction)),
            macro_f1=float(f1_score(truth, prediction, average='macro', zero_division=0)),
            class_labels=labels.tolist(),
            confusion_matrix=confusion_matrix(truth, prediction, labels=labels).tolist(),
            per_class=[dict(label=int(label), support=int(support[i]),
                            predicted=int(np.sum(prediction == label)),
                            precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]))
                       for i, label in enumerate(labels)])
    else:
        report.update(mae=float(np.mean(np.abs(truth-prediction))),
                      rmse=float(np.sqrt(np.mean((truth-prediction)**2))))
    return report


def cached_recipe_evidence(search, recipes, recipe_index, metric, max_chars=1800):
    index = int(recipe_index)
    if not 0 <= index < len(recipes):
        raise ValueError('recipe index out of range')

    def identity(recipe):
        return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()

    recipe = recipes[index]
    recipe_id = identity(recipe)
    incumbent_recipe = search.best['recipe'] if search.best else None
    incumbent_index = next((i for i, value in enumerate(recipes)
                            if identity(value) == identity(incumbent_recipe)), None)
    records = [record for record in search.records if identity(record['recipe']) == recipe_id]
    valid = sorted((record for record in records if record.get('valid')),
                   key=lambda record: (-float(record['score']), record['id']))
    result = dict(
        recipe_index=index, recipe_id=recipe_id, recipe=recipe,
        venue=search.protocol, metric=metric, cache_only=True, no_additional_fit=True,
        matches_incumbent_recipe=(incumbent_index == index),
        incumbent=dict(
            candidate_id=search.best['id'] if search.best else None,
            recipe_index=incumbent_index, recipe=incumbent_recipe,
            score=float(search.best['score']) if search.best else None,
            config=search.best['config'] if search.best else None),
        best=(dict(candidate_id=valid[0]['id'], score=float(valid[0]['score']),
                   config=valid[0]['config']) if valid else None),
        top_configs=[dict(candidate_id=record['id'], score=float(record['score']),
                          config=record['config']) for record in valid[:3]],
        failed_count=sum(not record.get('valid') for record in records),
        total_count=len(records))
    while len(json.dumps(result, ensure_ascii=False, default=str)) > max_chars and result['top_configs']:
        result['top_configs'].pop()
    if len(json.dumps(result, ensure_ascii=False, default=str)) > max_chars:
        result['recipe'] = {'recipe_id': recipe_id, 'detail': 'omitted from bounded tool output'}
        if result['best']:
            result['best']['config'] = {'detail': 'omitted from bounded tool output'}
        result['incumbent']['config'] = {'detail': 'omitted from bounded tool output'}
        result['incumbent']['recipe'] = {
            'recipe_id': identity(incumbent_recipe), 'detail': 'omitted from bounded tool output'}
    if len(json.dumps(result, ensure_ascii=False, default=str)) > max_chars:
        raise ValueError('cached recipe evidence cannot fit the bounded tool output')
    return result
