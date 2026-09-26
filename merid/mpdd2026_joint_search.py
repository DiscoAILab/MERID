"""Budgeted joint search over the MPDD-2026 binary/PHQ and hierarchical ternary heads."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from copy import deepcopy
import numpy as np

try:
    from .space import (DEFAULT, SEARCH_AXES, SPACE, TERNARY_DEFAULT, TERNARY_SPACE,
                        free_axes, pinned_axes)
except ImportError:
    from space import (DEFAULT, SEARCH_AXES, SPACE, TERNARY_DEFAULT, TERNARY_SPACE,
                       free_axes, pinned_axes)

METRIC_CONTRACT = 'mpdd2026_raw_phq_track'
FAMILIES = ('binary_phq', 'hierarchical_ternary')
HIER_FIXED = {'sev_base_mode': 'stats_q_pz'}


def _choices(space, axis):
    return list(space[axis][1])


def validate_envelope(value, pinned=None):
    if not isinstance(value, dict) or set(value) != {'head_family', 'parameters'}:
        raise ValueError('joint proposal must contain exactly head_family and parameters')
    family, parameters = value['head_family'], value['parameters']
    if family not in FAMILIES or not isinstance(parameters, dict):
        raise ValueError('joint proposal has an unsupported head family or parameters')
    if any(isinstance(v, bool) for v in parameters.values()):
        raise ValueError('head parameters do not accept boolean aliases for numeric choices')
    if family == 'binary_phq':
        if set(parameters) != set(SPACE) or any(parameters[key] not in _choices(SPACE, key)
                                                for key in SPACE):
            raise ValueError('binary_phq parameters must exactly match the bounded PHQ space')
    else:
        required = set(TERNARY_SPACE) | set(HIER_FIXED)
        if (set(parameters) != required
                or any(parameters[key] not in _choices(TERNARY_SPACE, key)
                       for key in TERNARY_SPACE)
                or any(parameters[key] != value for key, value in HIER_FIXED.items())):
            raise ValueError('hierarchical parameters must exactly match the bounded ternary space')
    for key, held in (pinned or {}).get(family, {}).items():
        if parameters.get(key) != held:
            raise ValueError('axis %r is held at %r by this search_axes setting, got %r'
                             % (key, held, parameters.get(key)))
    return {'head_family': family, 'parameters': deepcopy(parameters)}


def parse_envelope(text, pinned=None):
    lines = [line.strip() for line in (text or '').splitlines() if line.strip()]
    if not lines:
        raise ValueError('joint proposal response is empty')
    payload = lines[-1]
    if lines[-1] == '```':
        fences = [i for i, line in enumerate(lines) if line.startswith('```')]
        if len(fences) != 2 or lines[fences[0]] not in ('```', '```json'):
            raise ValueError('joint proposal must have one unambiguous final JSON code block')
        payload = '\n'.join(lines[fences[0] + 1:-1])
    try:
        return validate_envelope(json.loads(payload), pinned)
    except json.JSONDecodeError as exc:
        raise ValueError('final nonempty response line must be the complete JSON envelope') from exc


class HeadProposer:
    def __init__(self, proposer, audit_path=None, search_axes='all'):
        if not callable(getattr(proposer, 'complete', None)):
            raise ValueError('HeadProposer requires an existing complete(system, prompt) proposer')
        self.proposer, self.last_reasoning = proposer, ''
        self.audit_path = audit_path
        self.total_api_calls, self.last_attempts = 0, []
        self.pinned, self.free = pinned_axes(search_axes), free_axes(search_axes)

    @property
    def traj(self):
        return getattr(self.proposer, 'traj', None)

    def complete(self, system, prompt, **kwargs):
        return self.proposer.complete(system, prompt, **kwargs)

    def propose(self, profile, history):
        compact_history = []
        for record in history[-20:]:
            compact_history.append({key: record[key] for key in (
                'step', 'id', 'head_family', 'choice', 'source', 'valid',
                'component_score', 'fit_calls', 'status') if key in record})
            if record.get('joint_evaluation'):
                compact_history[-1]['joint_track'] = record['joint_evaluation']['track']
        prompt = (
            'Propose the NEXT NEW component experiment to evaluate, not the current winner. '
            'Previously evaluated components are already cached: never return a choice in HISTORY. '
            'The objective is the joint track=(best binary Macro-F1 + raw PHQ CCC '
            '+ best hierarchical ternary Macro-F1)/3. Binary/PHQ component_score sums two '
            'metrics while hierarchical component_score is one metric; their raw scores are '
            'not directly comparable across families. Seek expected improvement to the cached '
            'joint score by exploring an unseen, affordable configuration. Both families can '
            'improve the result; do not keep proposing the family with the larger current sum. '
            'binary_rule selects where the binary cut on the predicted PHQ score comes from: '
            '"oof_tuned" fits it on the development out-of-fold, "label_cut" uses the cut this '
            'corpus\'s binary label is defined by. Both are seeded and already measured; propose a '
            'value for it deliberately rather than copying the seed. '
            'In gp_stats_pz, audio/video extractor aliases do not add audio/video feature blocks; '
            'however extractor-dependent pair availability/counts can still differ. '
            'Do not assume feature equivalence from mode names alone. '
            'The fixed metric contract is '
            f'{METRIC_CONTRACT}. Candidate families are binary_phq with parameters '\
            f'{json.dumps(self.free["binary_phq"])}, or '
            'hierarchical_ternary with parameters '
            f'{json.dumps({**self.free["hierarchical_ternary"], **HIER_FIXED})}. '
            + (f'The remaining axes are HELD FIXED for this run and must be returned exactly '
               f'as given: {json.dumps(self.pinned)}. A proposal that moves one is rejected. '
               if self.pinned else '') +
            'Use the shared fixed-fold development protocol. Output the complete JSON envelope '
            '{"head_family":...,"parameters":...} on the final nonempty line.\n'
            'PROFILE:\n' + json.dumps(profile, default=str) + '\nHISTORY:\n' +
            json.dumps(compact_history, default=str))
        seen = {_key(row['choice']) for row in history if 'choice' in row}
        self.last_attempts = []
        for attempt in range(2):
            self.total_api_calls += 1
            entry = dict(api_call=self.total_api_calls, attempt=attempt + 1,
                         completed_candidates=len(history))
            try:
                text = self.proposer.complete('You propose the next unseen MPDD2026 experiment.',
                                              prompt, max_tokens=1600)
                self.last_reasoning = text or ''
                entry['response'] = self.last_reasoning
                choice = parse_envelope(text, self.pinned)
                duplicate = _key(choice) in seen
                entry.update(choice=choice, status='duplicate' if duplicate else 'accepted')
            except Exception as exc:
                entry.update(status=type(exc).__name__)
                raise
            finally:
                self.last_attempts.append(entry)
                if self.audit_path is not None:
                    with open(self.audit_path, 'a', encoding='utf-8') as audit:
                        audit.write(json.dumps(entry, ensure_ascii=False) + '\n')
            if not duplicate:
                return choice
            prompt += ('\nREJECTED: this choice was already evaluated and is cached: '
                       + json.dumps(choice) + '. Return a different unseen configuration. '
                       'You are choosing the next experiment, not reporting the best known model.')
        raise ValueError('LLM did not propose an unseen component after two audited attempts')


class JointRandomProposer:
    def __init__(self, seed=0, search_axes='all'):
        self.rng, self.last_reasoning = random.Random(seed), ''
        self.pinned, self.free = pinned_axes(search_axes), free_axes(search_axes)

    def propose(self, profile, history):
        family = self.rng.choice(FAMILIES)
        space = SPACE if family == 'binary_phq' else TERNARY_SPACE
        parameters = {key: self.rng.choice(_choices(space, key)) for key in self.free[family]}
        parameters.update(self.pinned.get(family, {}))
        if family == 'hierarchical_ternary':
            parameters.update(HIER_FIXED)
        return validate_envelope({'head_family': family, 'parameters': parameters}, self.pinned)


def _key(choice):
    return json.dumps(choice, sort_keys=True, separators=(',', ':'))


def _candidate_id(choice):
    return hashlib.sha256(_key(choice).encode()).hexdigest()


def _validate_result(result, family, harness):
    if not isinstance(result, dict):
        raise ValueError('component evaluator must return an evidence mapping')
    protocol = harness.fold_protocol
    ids = [str(value) for value in result.get('patient_ids', [])]
    if (result.get('data_scope') != 'development' or result.get('nested_final_estimate') is not False
            or type(result.get('completed_folds')) is not int
            or result.get('completed_folds') != harness.n_folds
            or ids != list(protocol.patient_ids)
            or result.get('protocol_fingerprint') != protocol.protocol_fingerprint
            or result.get('protocol_evidence_fingerprint') != protocol.evidence_fingerprint
            or result.get('phq_metric_contract') != METRIC_CONTRACT):
        raise ValueError('component evidence does not match the shared metric/fold protocol')
    fit_calls = result.get('fit_calls')
    expected_cost = harness.n_folds * (1 if family == 'binary_phq' else 6)
    if type(fit_calls) is not int or fit_calls != expected_cost:
        raise ValueError('completed component fit cost must exactly match its declared fold/seed contract')

    def metric(key, low=0., high=1.):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('invalid component metric: ' + key)
        return float(value)

    if family == 'binary_phq':
        prediction = np.asarray(result.get('phq_oof'), dtype=float)
        if (prediction.shape != (len(ids),) or not np.isfinite(prediction).all()
                or np.any((prediction < 0.) | (prediction > 27.))):
            raise ValueError('PHQ OOF must be a complete finite 0..27 vector')
        score = metric('binF1') + metric('ccc', -1., 1.)
        expected_track = (score + metric('terF1')) / 3.
        if not math.isclose(metric('track', -1./3., 1.), expected_track, abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError('PHQ reference track disagrees with its component metrics')
    else:
        score = metric('terF1')
        for key in ('gate_fit_calls', 'severity_fit_calls'):
            if type(result.get(key)) is not int or result[key] != harness.n_folds * 3:
                raise ValueError('hierarchical gate/severity fit costs disagree with folds/seeds')
        for key in ('gate_oof', 'severity_oof'):
            probabilities = np.asarray(result.get(key), dtype=float)
            if (probabilities.shape != (len(ids), 2) or not np.isfinite(probabilities).all()
                    or np.any((probabilities < 0.) | (probabilities > 1.))
                    or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-8, rtol=0.)):
                raise ValueError('hierarchical OOF must contain complete normalized probability rows')
    return score, fit_calls


def _joint(binary_record, ternary_record):
    binary, ternary = binary_record['evaluation'], ternary_record['evaluation']
    track = (float(binary['binF1']) + float(binary['ccc']) + float(ternary['terF1'])) / 3.0
    return {
        'track': track, 'binF1': float(binary['binF1']), 'ccc': float(binary['ccc']),
        'terF1': float(ternary['terF1']), 'phq_metric_contract': METRIC_CONTRACT,
        'protocol_fingerprint': binary['protocol_fingerprint'],
        'protocol_evidence_fingerprint': binary['protocol_evidence_fingerprint'],
        'patient_ids': list(binary['patient_ids']), 'nested_final_estimate': False,
        'cache_only_recombination': True,
        'component_ids': {'binary_phq': binary_record['id'],
                          'hierarchical_ternary': ternary_record['id']},
        'component_choices': {'binary_phq': deepcopy(binary_record['choice']),
                              'hierarchical_ternary': deepcopy(ternary_record['choice'])},
        'phq_derived_3head_reference': {
            'candidate_id': binary_record['id'], 'track': float(binary['track']),
            'terF1': float(binary['terF1'])},
    }


def label_cut_evidence(harness, lo=0.0, hi=20.0, step=0.5):
    phq = getattr(harness, 'phq', None)
    labels = (getattr(harness, 'labels', None) or {}).get('binary')
    if not phq or not labels:
        return None
    ids = sorted(set(phq) & set(labels))
    if len(ids) < 20:
        return None
    truth = [int(labels[i]) for i in ids]
    best = None
    cut = lo
    while cut <= hi + 1e-9:
        agree = sum(1 for i, t in zip(ids, truth) if int(float(phq[i]) >= cut) == t) / len(ids)
        if best is None or agree > best[1] + 1e-12:
            best = (cut, agree)
        cut += step
    return {'best_cut': float(best[0]), 'agreement': round(float(best[1]), 4), 'n_patients': len(ids),
            'reading': ('the binary label IS the PHQ score thresholded at this cut, so "label_cut" '
                        'deploys the definition itself and does not depend on the fold partition'
                        if best[1] >= 0.999 else
                        'the label is not exactly this threshold, so a fitted cut may differ for a reason')}


def _review_evidence(history, leaders, best, profile, budget, fit_budget, used_fits, harness):
    def candidate(record):
        return dict(id=record.get('id'), step=record.get('step'),
                    head_family=record.get('head_family'), source=record.get('source'),
                    valid=bool(record.get('valid')), component_score=record.get('component_score'),
                    fit_calls=record.get('fit_calls'), status=record.get('status'),
                    parameters=(record.get('choice') or {}).get('parameters'))

    scored = [candidate(r) for r in history]
    failures = {}
    for record in history:
        if not record.get('valid') and record.get('status'):
            failures[record['status']] = failures.get(record['status'], 0) + 1
    constraints = dict(
        corpus='MPDD-2026', metric_contract=METRIC_CONTRACT,
        head_families=list(FAMILIES),
        parameter_space={'binary_phq': {k: list(v[1]) for k, v in SPACE.items()},
                         'hierarchical_ternary': {k: list(v[1]) for k, v in TERNARY_SPACE.items()}},
        fixed_hierarchical_axes=dict(HIER_FIXED),
        venue='cross-validated folds of the training split; the official evaluation is not run here '
              'and is not visible to this review',
        search_contract=dict(
            promotion='A component replaces its family leader only on a strict improvement in the '
                      'component score; the joint track score is recomputed from the two leaders.',
            revision_capabilities=['clf_space'],
            capability_meaning='On this corpus a revision cannot edit an adapter, a recipe space or '
                               'a metric - none of those exist here. The single permitted action is '
                               'REVISE: clf_space, which means CONTINUE THE SEARCH in a direction '
                               'you state: name the head family and the axis or value region to try '
                               'that the history shows was never scored. Your note is handed to the '
                               'proposer verbatim as that direction.',
            unsupported_revisions='exclude_source, recipe_space, adapter and eval_metric do not '
                                  'apply on this corpus; requesting one records a limitation and '
                                  'changes nothing.'),
        budget=dict(candidate_budget=budget, candidates_used=len(history),
                    fit_budget=fit_budget, fit_calls_used=used_fits,
                    folds=int(getattr(harness, 'n_folds', 0))))
    shared = dict(immutable_constraints=constraints, corpus='MPDD-2026',
                  primary_metric='track', venue=constraints['venue'],
                  component_leaders={family: dict(id=record.get('id'),
                                                  component_score=record.get('component_score'),
                                                  parameters=(record.get('choice') or {}).get('parameters'))
                                     for family, record in leaders.items()},
                  joint_best=dict(track=(best or {}).get('track'),
                                  component_ids=(best or {}).get('component_ids')),
                  remaining_candidates=max(0, budget - len(history)),
                  revision_policy='Add candidates; only a strict improvement in a component score '
                                  'moves that family leader. The metric is fixed.')
    evidence = {role: dict(shared) for role in ('planner', 'engineer', 'reviewer')}
    evidence['planner'].update(
        regime_profile={k: v for k, v in (profile or {}).items()
                        if k in ('summary', 'hints', 'binary_label_provenance')},
        seeds_were_reference_components=any(r.get('source') == 'seed_reference' for r in history))
    evidence['engineer'].update(
        failed_candidates=failures, n_failed=sum(failures.values()),
        fit_accounting=constraints['budget'])
    evidence['reviewer'].update(candidates=scored, axis_coverage=_axis_coverage(scored))
    return evidence


def _axis_coverage(scored):
    out = {}
    for family, space in (('binary_phq', SPACE), ('hierarchical_ternary', TERNARY_SPACE)):
        rows = [c for c in scored if c['head_family'] == family and c.get('parameters')]
        per_axis = {}
        for axis, (_kind, choices) in space.items():
            tried = {c['parameters'].get(axis) for c in rows}
            per_axis[axis] = dict(
                scored=[v for v in choices if v in tried],
                never_scored=[v for v in choices if v not in tried])
        out[family] = dict(n_candidates=len(rows), axes=per_axis)
    return out


def _hold_joint_review(proposer, evidence, budget_checks, traj, ctx, prior_rounds, mode):
    try:
        from .retrospective import hold_review
    except ImportError:
        from retrospective import hold_review
    reviewer_tools = {'reviewer': {}, 'planner': {}, 'engineer': {}}
    return hold_review(proposer, evidence, tools=reviewer_tools, prior_rounds=prior_rounds,
                       budget=budget_checks, traj=traj, ctx=ctx, return_status=True, mode=mode)


def optimize_joint(harness, proposer, budget=12, seed=42, log_path=None, profile=None,
                   fit_budget=720, reference_components=None, retrospective=0, review_checks=2,
                   review_mode='roles', revision_reserve=0.25, traj=None, ctx='MPDD-2026',
                   search_axes='all'):
    if type(budget) is not int or budget < 2:
        raise ValueError('joint_heads requires candidate budget >= 2 for both seed components')
    if type(fit_budget) is not int or fit_budget < 1:
        raise ValueError('fit_budget must be a positive integer')
    if search_axes not in SEARCH_AXES:
        raise ValueError('search_axes must be one of %s' % (SEARCH_AXES,))
    pinned = pinned_axes(search_axes)
    references = {} if reference_components is None else reference_components
    if not isinstance(references, dict) or set(references) - set(FAMILIES):
        raise ValueError('reference_components must map supported head families to envelopes')
    references = {family: validate_envelope(value, pinned)
                  for family, value in references.items()}
    if any(value['head_family'] != family for family, value in references.items()):
        raise ValueError('reference component envelope family must match its mapping key')
    if log_path and os.path.exists(log_path) and os.path.getsize(log_path):
        raise ValueError('joint_heads does not resume an existing trajectory')
    folds = int(harness.n_folds)
    required_seed_fits = folds + folds * 6
    if fit_budget < required_seed_fits:
        raise ValueError('fit budget cannot establish both required seed components')
    if type(proposer).__name__ == 'RandomProposer':
        proposer = JointRandomProposer(seed, search_axes)
    rng, seen, history, leaders = random.Random(seed), set(), [], {}
    evidence = label_cut_evidence(harness)
    if evidence and profile is not None:
        profile = dict(profile)
        profile['binary_label_provenance'] = evidence
    best, used_fits, stop_reason = None, 0, None
    seeds = [
        {'head_family': 'binary_phq', 'parameters': dict(DEFAULT)},
        {'head_family': 'hierarchical_ternary',
         'parameters': {**TERNARY_DEFAULT, **HIER_FIXED}},
    ]
    seeds = [references.get(choice['head_family'], choice) for choice in seeds]
    logf = open(log_path, 'a', encoding='utf-8') if log_path else None
    rounds = max(0, int(retrospective))
    extra_candidates = int(math.ceil(budget * max(0.0, float(revision_reserve)))) if rounds else 0
    extra_fits = int(math.ceil(fit_budget * max(0.0, float(revision_reserve)))) if rounds else 0
    reviews = []
    try:
        for review_round in range(rounds + 1):
            while len(history) < budget:
                if len(history) < 2:
                    choice, source, reasoning = seeds[len(history)], 'seed', 'deterministic required seed'
                    if choice['head_family'] in references:
                        source, reasoning = 'seed_reference', 'explicit reference component; refitted within the seed budget'
                else:
                    proposal_profile = dict(profile or {})
                    proposal_profile['joint_search'] = {
                        'candidate_budget': budget, 'candidates_used': len(history),
                        'fit_budget': fit_budget, 'fit_calls_used': used_fits,
                        'component_incumbents': {family: {
                            'candidate_id': record['id'], 'component_score': record['component_score']}
                            for family, record in leaders.items()}}
                    try:
                        choice, source = proposer.propose(proposal_profile, history), 'llm'
                        reasoning = getattr(proposer, 'last_reasoning', '')
                        choice = validate_envelope(choice, pinned)
                    except Exception as exc:
                        choice = JointRandomProposer(rng.randrange(2 ** 32),
                                                     search_axes).propose({}, history)
                        source = 'fallback'
                        reasoning = '[joint manager] proposal failed: ' + type(exc).__name__
                    if type(proposer).__name__ == 'JointRandomProposer':
                        source = 'random'
                    if _key(choice) in seen:
                        for _ in range(500):
                            fallback = JointRandomProposer(rng.randrange(2 ** 32),
                                                           search_axes).propose({}, history)
                            if _key(fallback) not in seen:
                                choice, source = fallback, 'random_duplicate_substitution'
                                break
                        else:
                            stop_reason = 'candidate_space_exhausted'; break
                estimated = folds if choice['head_family'] == 'binary_phq' else folds * 6
                if used_fits + estimated > fit_budget:
                    stop_reason = 'insufficient_remaining_fit_budget'; break
                seen.add(_key(choice))
                record = dict(step=len(history) + 1, id=_candidate_id(choice), choice=choice,
                              head_family=choice['head_family'], source=source, reasoning=reasoning,
                              valid=False, fit_calls=0)
                if source == 'llm':
                    record['proposal_attempts'] = deepcopy(getattr(proposer, 'last_attempts', []))
                returned_cost = None
                result_returned = False
                try:
                    result = (harness.evaluate(choice['parameters']) if choice['head_family'] == 'binary_phq'
                              else harness.evaluate_ternary(choice['parameters']))
                    result_returned = True
                    if isinstance(result, dict) and type(result.get('fit_calls')) is int and result['fit_calls'] >= 0:
                        returned_cost = result['fit_calls']
                    score, actual = _validate_result(result, choice['head_family'], harness)
                    record.update(valid=True, component_score=score, fit_calls=actual,
                                  evaluation=result)
                    incumbent = leaders.get(choice['head_family'])
                    if incumbent is None or score > incumbent['component_score'] + 1e-12:
                        leaders[choice['head_family']] = record
                    if len(leaders) == 2:
                        combined = _joint(leaders['binary_phq'], leaders['hierarchical_ternary'])
                        record['joint_evaluation'] = combined
                        if best is None or combined['track'] > best['track'] + 1e-12:
                            best = {'track': combined['track'], 'source': 'joint_cached_recombination',
                                    'choice': deepcopy(combined['component_choices']),
                                    'component_ids': deepcopy(combined['component_ids']),
                                    'evaluation': combined}
                except Exception as exc:
                    actual = (returned_cost if returned_cost is not None else
                              getattr(exc, 'fit_calls', getattr(exc, 'attempted_fit_calls', 0)))
                    if type(actual) is not int or actual < 0:
                        actual = 0
                    record.update(status=type(exc).__name__, fit_calls=actual)
                used_fits += record['fit_calls']
                history.append(record)
                if logf:
                    logf.write(json.dumps(record, ensure_ascii=False) + '\n'); logf.flush()
                if result_returned and returned_cost is None:
                    raise RuntimeError('component returned unverifiable fit cost; refusing a completed budget claim')
                if used_fits > fit_budget:
                    raise RuntimeError('actual component fit cost exceeded the declared fit budget')
            if review_round >= rounds or stop_reason == 'candidate_space_exhausted':
                break
            if len(leaders) != 2:
                break
            if not hasattr(proposer, 'complete'):
                reviews.append(dict(round=review_round + 1, status=dict(
                    state='review_unavailable', complete=False,
                    reason='proposer has no complete(); a review needs a language model'),
                    decision=None, minutes='', candidates_before=len(history)))
                break
            evidence = _review_evidence(history, leaders, best, profile, budget, fit_budget,
                                        used_fits, harness)
            verdict, minutes, status = _hold_joint_review(
                proposer, evidence, review_checks, traj, ctx,
                [r['status'] for r in reviews] or None, review_mode)
            reviews.append(dict(round=review_round + 1, status=status, decision=verdict,
                                minutes=minutes, candidates_before=len(history)))
            if verdict is None:
                break
            if verdict['step'] != 'clf_space':
                reviews[-1]['outcome'] = 'unsupported_revision_on_this_corpus'
                break
            budget += extra_candidates
            fit_budget += extra_fits
            profile = dict(profile or {})
            profile['review_direction'] = verdict['note']
            reviews[-1]['outcome'] = 'search_extended'
            reviews[-1]['budget_after'] = dict(candidate_budget=budget, fit_budget=fit_budget)

    finally:
        if logf:
            logf.close()
    if len(leaders) != 2:
        raise RuntimeError('joint search ended without valid binary/PHQ and hierarchical components')
    best['fit_budget'] = {'limit': fit_budget, 'used': used_fits,
                          'stop_reason': stop_reason or 'candidate_budget_complete'}
    best['review'] = dict(
        rounds_held=len(reviews), review_mode=review_mode,
        revisions_proposed=sum(1 for r in reviews if r.get('decision')),
        revisions_executed=sum(1 for r in reviews if r.get('outcome') == 'search_extended'),
        reserve_candidates=extra_candidates, reserve_fits=extra_fits,
        records=[{k: v for k, v in r.items() if k != 'minutes'} for r in reviews],
        minutes=[r.get('minutes') for r in reviews])
    return history, best
