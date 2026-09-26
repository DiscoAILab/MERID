"""The self-improvement loop: coupled search, evidence-guided review rounds, revisions and inheritance of the retained pipeline."""
from __future__ import annotations

import itertools
import json
import math
import hashlib
import random
import re
from pathlib import Path

import numpy as np

try:
    from .development_search import DevelopmentSearch, CheckpointPersistenceError, fingerprint
    from .reliability import assess_search
    from .development_diagnostics import (cached_recipe_evidence, explored_surface,
                                          selection_diagnostics)
    from .adapter_source_evidence import AdapterSourceEvidence
    from .retrospective import hold_review, write_minutes, ROLE_SYSTEM, CHAIR_SYSTEM
    from .research_memory import ResearchMemory, MEMORY_USE
except ImportError:
    from development_search import DevelopmentSearch, CheckpointPersistenceError, fingerprint
    from reliability import assess_search
    from development_diagnostics import (cached_recipe_evidence, explored_surface,
                                         selection_diagnostics)
    from adapter_source_evidence import AdapterSourceEvidence
    from retrospective import hold_review, write_minutes, ROLE_SYSTEM, CHAIR_SYSTEM
    from research_memory import ResearchMemory, MEMORY_USE


DEFAULT_REVISION_RESERVE = 0.25


def _union(old, new):
    out, seen = [], set()
    for value in list(old) + list(new):
        key = fingerprint(value)
        if key not in seen:
            out.append(value)
            seen.add(key)
    return out


RANDOM_REVISION_DRAW = 4


def _value_product(items):
    keys = sorted({k for item in items for k in item})
    values = [sorted({json.dumps(item.get(k), sort_keys=True) for item in items}) for k in keys]
    return [{k: json.loads(v) for k, v in zip(keys, combo)} for combo in itertools.product(*values)]


def random_revision(usable, configs, seed, round_no, draw=RANDOM_REVISION_DRAW):
    rng = random.Random('random-revision:%s:%s' % (seed, round_no))
    have = {fingerprint(item) for item in list(usable) + [c for c in configs if isinstance(c, dict)]}
    recipes = [r for r in _value_product(usable)
               if fingerprint(r) not in have and any(v for k, v in r.items() if k != 'agg')]
    families = {}
    for config in configs:
        if isinstance(config, dict) and config.get('model'):
            families.setdefault(config['model'], []).append(config)
    pool = []
    for family in sorted(families):
        pool.extend(c for c in _value_product(families[family]) if fingerprint(c) not in have)
    recipes.sort(key=fingerprint)
    pool.sort(key=fingerprint)
    if recipes and (not pool or rng.random() < 0.5):
        step, candidates = 'recipe_space', recipes
    elif pool:
        step, candidates = 'clf_space', pool
    else:
        return None
    return step, rng.sample(candidates, min(draw, len(candidates)))


def _neutral_recipe(usable, references, arm):
    reference_keys = {fingerprint(item['recipe']) for item in references}
    for recipe in usable:
        if fingerprint(recipe) not in reference_keys:
            return recipe
    raise RuntimeError('%s has no neutral recipe to hold its axis at: every usable recipe came '
                       'from a carried-over reference' % arm)


def _neutral_probe(corpus, all_configs, references, arm):
    probe = getattr(corpus, '_probe_cfg', None)
    if callable(probe):
        return probe()
    reference_keys = {fingerprint(item['config']) for item in references}
    for config in all_configs:
        if config != 'codegen' and fingerprint(config) not in reference_keys:
            print('  [%s] no corpus probe estimator; holding at the first NON-reference config'
                  % arm, flush=True)
            return config
    raise RuntimeError('%s has no neutral value to hold its axis at: the corpus exposes no probe '
                       'estimator and every candidate config came from a carried-over reference'
                       % arm)


def _authoritative_ranking(search, metric, max_other_candidates=3, tolerance=1e-12):
    if search.best is None:
        raise ValueError('authoritative ranking requires a validated incumbent')
    if (type(max_other_candidates) is not int or max_other_candidates < 0
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('authoritative ranking bounds must be finite and nonnegative')
    valid = [record for record in search.records if record.get('valid')]
    ids = [record.get('id') for record in valid]
    scores = [float(record['score']) for record in valid]
    if (any(candidate_id is None for candidate_id in ids) or len(ids) != len(set(ids))
            or not all(np.isfinite(score) for score in scores)):
        raise ValueError('valid ranking records require unique IDs and finite scores')
    incumbent_matches = [record for record in valid if record['id'] == search.best.get('id')]
    if len(incumbent_matches) != 1:
        raise ValueError('authoritative incumbent must identify exactly one valid record')
    ranked = sorted(valid, key=lambda record: (-float(record['score']), str(record['id'])))
    best_score = float(search.best['score'])
    promotion_se = float(getattr(search, 'promotion_se', 0.0) or 0.0)
    rule = getattr(search, 'selection_rule', 'incumbent')
    if rule == 'one_se_of_best':
        band = float((getattr(search, 'selected_1se', None) or {}).get('se', 0.0) or 0.0)
        incumbent_se, margin = band, max(tolerance, band)
    else:
        incumbent_se = float(search.best.get('se', 0.0) or 0.0) if promotion_se > 0 else 0.0
        margin = max(tolerance, promotion_se * incumbent_se) if promotion_se > 0 else tolerance
    if (abs(float(incumbent_matches[0]['score']) - best_score) > tolerance
            or float(ranked[0]['score']) > best_score + margin):
        raise ValueError('authoritative incumbent is inconsistent with executed valid records')
    tied = [record for record in valid
            if abs(float(record['score']) - best_score) <= tolerance]
    distinct = [float(record['score']) for record in valid
                if float(record['score']) < best_score - tolerance]
    retained_above = [record for record in valid if float(record['score']) > best_score + tolerance]
    others = [record for record in ranked if record['id'] != search.best['id']]

    def identity(record):
        return dict(candidate_id=record['id'], recipe=record['recipe'],
                    config=record['config'], score=float(record['score']))

    return dict(
        venue=search.protocol,
        metric=metric,
        valid_candidate_count=len(valid),
        incumbent_candidate_id=search.best['id'],
        best_score=best_score,
        tied_best_count=len(tied),
        tie_tolerance=tolerance,
        top_other_candidates=[identity(record) for record in others[:max_other_candidates]],
        next_distinct_score=max(distinct) if distinct else None,
        promotion_rule=(
            ('the simplest candidate within one subject-bootstrap SE of the BEST score is selected '
             '(SE %.4f); %d executed candidate(s) score above it inside that band, which the rule '
             'intends - a higher score inside one SE is not a better candidate'
             % (incumbent_se, len(retained_above))) if rule == 'one_se_of_best' else
            ('incumbent retained unless a challenger clears %g subject-bootstrap SE '
             '(SE %.4f, margin %.4f); %d executed candidate(s) score above the incumbent '
             'but within that margin and were therefore NOT promoted'
             % (promotion_se, incumbent_se, margin, len(retained_above)))
            if promotion_se > 0 else 'strict improvement > 1e-12'),
        higher_scoring_retained=[identity(record) for record in retained_above[:max_other_candidates]],
        uncertainty=('Incumbent subject-bootstrap SE = %.4f on this venue.' % incumbent_se)
        if incumbent_se > 0 and (promotion_se > 0 or rule == 'one_se_of_best') else
        'No confidence interval or standard error was computed.')


def _reference_candidates(args):
    source = getattr(args, 'reference_candidates', None)
    if not source:
        return [], None
    path = Path(source).resolve()
    raw = path.read_bytes()
    payload = json.loads(raw.decode('utf-8'))
    if not isinstance(payload, list) or not payload or len(payload) > 64:
        raise ValueError('reference_candidates must be a nonempty JSON list of at most 64 pairs')
    if len(payload) > args.max_evaluations:
        raise ValueError('reference candidates exceed max_evaluations')
    out, seen = [], set()
    for item in payload:
        if not isinstance(item, dict) or set(item) != {'recipe', 'config'}:
            raise ValueError('each reference candidate must contain exactly recipe and config')
        if not isinstance(item['recipe'], dict) or not isinstance(item['config'], dict):
            raise ValueError('reference recipe and config must be JSON objects')
        key = fingerprint(item)
        if key in seen:
            raise ValueError('duplicate reference candidate pair')
        seen.add(key)
        out.append(item)
    return out, {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                 'count': len(out)}


class _ConstrainedReviewProposer:
    def __init__(self, proposer, constraint_text):
        self.proposer, self.constraint_text = proposer, constraint_text

    def complete(self, system, prompt, **kwargs):
        return self.proposer.complete(system, self.constraint_text + '\n\n' + prompt, **kwargs)


def apply_reliability_gate(search, args, log=print):
    if str(getattr(args, 'reliability_gate', 'on')).lower() == 'off':
        return None
    try:
        from .reliability import assess_promotion
    except ImportError:
        from reliability import assess_promotion

    gate = {}
    try:
        transfer = assess_search(search)
    except Exception as exc:
        transfer = {'error': type(exc).__name__ + ': ' + str(exc)[:200]}
    if transfer:
        gate.update(transfer)
        if transfer.get('usable') is not None:
            log('  [reliability] development ranking %s (worst rho %+.3f over %d folds) - '
                'reported, not acted on'
                % ('transfers' if transfer['usable'] else 'does not transfer',
                   transfer['worst_rho'], transfer['n_folds']))

    try:
        promotion = assess_promotion(search)
    except Exception as exc:
        gate['promotion'] = {'error': type(exc).__name__ + ': ' + str(exc)[:200]}
        return gate or None
    if promotion:
        gate['promotion'] = promotion
        if promotion.get('applicable') and not promotion['cleared']:
            reference = next(r for r in search.records if r['id'] == promotion['reference_id'])
            search.best = reference
            search.selected_model = None
            gate['action'] = 'reverted to the reference: the promotion did not clear the ' \
                             "winner's-curse margin"
            gate['selection_changed'] = True
            log("  [reliability] promotion led by %+.4f over %d candidates but the winner's-curse "
                'margin is %.4f (%.2f x SE %.4f) -> reverted to the reference'
                % (promotion['lead'], promotion['n_candidates'], promotion['margin'],
                   promotion['margin'] / promotion['reference_se'] if promotion['reference_se']
                   else float('nan'), promotion['reference_se']))
        elif promotion.get('applicable'):
            gate['action'] = 'promotion cleared the winner\'s-curse margin'
            gate['selection_changed'] = False
            log("  [reliability] promotion led by %+.4f over %d candidates; margin %.4f -> kept"
                % (promotion['lead'], promotion['n_candidates'], promotion['margin']))
        else:
            gate['action'] = 'deployed candidate is a reference; nothing to defend'
            gate['selection_changed'] = False
    return gate or None


def run_development(corpus, args, proposer, load, recipes, report, extractors,
                    adapter_reasoning, traj, adapter_source=None):
    memory_mode = getattr(args, 'research_memory', 'off')
    if memory_mode not in ('off', 'evolving'):
        raise ValueError('research_memory must be off or evolving')
    if memory_mode == 'evolving' and args.retrospective < 1:
        raise ValueError('evolving research memory requires at least one review round')
    if getattr(args, 'review_mode', 'roles') == 'random' and memory_mode != 'off':
        raise ValueError('the random-revision arm has no reasoning to remember; research_memory must be off')
    accept_policy = getattr(args, 'review_accept_policy', 'stop') or 'stop'
    if accept_policy not in ('stop', 'require_revision'):
        raise ValueError('review_accept_policy must be stop or require_revision')
    kind = 'regression' if corpus._is_reg() else 'classification'
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reserve = getattr(args, 'revision_reserve', None)
    reserve = DEFAULT_REVISION_RESERVE if reserve is None else float(reserve)
    held_back = int(math.ceil(args.max_evaluations * max(0.0, reserve)))
    total_budget = args.max_evaluations + held_back
    reserve_schedule = getattr(args, 'reserve_schedule', 'front') or 'front'
    if reserve_schedule not in ('front', 'equal'):
        raise ValueError('reserve_schedule must be front or equal')
    search = DevelopmentSearch(load, task=corpus.TASK, kind=kind, builder=corpus._make,
                               scorer=corpus._dev, seed=args.seed,
                               n_splits=args.development_folds,
                               max_evaluations=total_budget,
                               promotion_se=float(getattr(args, 'promotion_se', 0.0) or 0.0),
                               selection_rule=getattr(args, 'selection_rule', 'incumbent'))
    # The revision reserve is held back from the initial search for the review rounds.
    if args.retrospective and held_back:
        search.max_evaluations = args.max_evaluations
    history, statuses = [], []
    memory = None
    source_evidence = AdapterSourceEvidence(adapter_source) if adapter_source else None
    if source_evidence:
        (output / 'adapter_snapshot.py').write_bytes(adapter_source.encode('utf-8'))

    def memory_status():
        if memory is None:
            return dict(mode=memory_mode, version=0)
        bank = memory.snapshot()
        return dict(mode=memory_mode, version=bank['version'], path=memory.path.name,
                    active_entries=sum(e['status'] == 'active' for e in bank['entries']),
                    updates=sum(e['state'] == 'updated' for e in bank['events']),
                    failures=sum(e['state'] == 'failed' for e in bank['events']),
                    llm_attempts=sum(e['llm_calls'] for e in bank['events']))

    def checkpoint(phase, run_status='incomplete'):
        agent_status = dict(
            backbone=getattr(corpus, 'backbone_name', 'run_corpus'),
            immutable_constraints=constraints,
            llm_calls=traj.info['model_stats']['api_calls'],
            completed_reviews=sum(s.get('complete', False) for s in statuses),
            review_statuses=statuses,
            revisions=history,
            research_memory=memory_status(),
        )
        if reference_source is not None:
            agent_status['reference_candidates'] = dict(
                reference_source, reevaluated=[{key: record[key] for key in
                    ('id', 'valid', 'score', 'error') if key in record}
                    for record in search.records if record.get('round') == -1])
        return search.checkpoint(output, phase=phase, agent_status=agent_status,
                                 run_status=run_status)
    references, reference_source = _reference_candidates(args)
    exclusions = {name.strip() for name in args.exclude_sources.split(',') if name.strip()}
    for candidate in references:
        if any(key != 'agg' and candidate['recipe'].get(key) for key in exclusions):
            raise ValueError('exact reference candidate enables an excluded source')
    recipes = _union([item['recipe'] for item in references], recipes)
    widths, usable = {}, []
    for recipe in recipes:
        try:
            d = search.view(recipe)
            widths[fingerprint(recipe)] = int(d['train_X'].shape[1])
            usable.append(recipe)
        except Exception as exc:
            traj.log('development_recipe_invalid', choice=recipe,
                     observation=type(exc).__name__ + ': ' + str(exc)[:300])
    if not usable:
        raise RuntimeError('No recipe satisfies the development identity/data contract: ' +
                           str(search.load_errors))
    first = search.view(usable[0])
    y = first['train_' + corpus.TASK]
    exclusions = sorted({name.strip() for name in args.exclude_sources.split(',') if name.strip()})
    provenance = [ln.strip() for ln in (report or '').splitlines()
                  if 'LABEL PROVENANCE' in ln or ln.strip().startswith('->')]
    constraints = dict(
        excluded_sources=exclusions,
        requested_task=corpus.TASK,
        policy='Immutable experimental condition. Excluded sources are deliberately disabled; '
               'their absence is not a loader failure. Do not ask to enable them, including '
               'equivalent renamed sources, in any proposal or revision.',
        prediction_evidence='Only nonexcluded observations available at prediction time may be '
                            'features or routing evidence. Target-defining measurements and '
                            'held-out labels are not prediction evidence. Training labels remain '
                            'permitted for fitting and development labels for the fixed metric.',
        measured_label_provenance=(provenance or ['The inspection did not compute label provenance for '
                                                  'this corpus (not a single table, or too few rows).']),
        loader_reads_fields=(source_evidence.field_reads() if source_evidence is not None
                             else 'adapter source not available to this run'),
        field_rule='The entry above lists every literal field name the executed loader reads. A field '
                   'that is NOT in it does not enter the features, whatever else its file contains: a '
                   'record that also holds scale totals or the label is not evidence that the loader '
                   'read them. Name a source target-defining only for a field the loader actually '
                   'reads, and say which field.',
        provenance_rule='The line above is a DIRECT MEASUREMENT of the target-defining question: it '
                        'reports how much of the label a threshold rule on one column, or on a simple '
                        'total of columns, reproduces on training rows. Where it says no column or '
                        'total reproduces the label, that is evidence AGAINST a target-defining '
                        'finding, and a verdict of exclude_source must then name a mechanism the '
                        'measurement does not cover (an item subset, a nonlinear scoring rule, a '
                        'documented derivation) rather than assert the label is thresholded from the '
                        'source. Where it names a column and its cut points, that IS the instrument '
                        'and the formula, and exclusion is mandatory.',
        evidence_interpretation='Separate measured facts from hypotheses. Aggregate scores alone '
                                'do not establish per-class recall, absence of leakage, a statistical '
                                'tie, or a difference measured in standard errors. Report per-class '
                                'recall, standard errors and confidence intervals only when actually '
                                'computed and supplied; otherwise state that they are unavailable. '
                                'A low score does not rule out leakage.',
        search_contract=dict(
            promotion=(('A complete candidate replaces the incumbent only when its selection score exceeds '
                        'the incumbent by more than %g subject-bootstrap standard error(s) of the incumbent '
                        'score. Anything closer is a tie, and ties retain the incumbent.'
                        % float(getattr(args, 'promotion_se', 0.0) or 0.0))
                       if float(getattr(args, 'promotion_se', 0.0) or 0.0) > 0 else
                       ('A complete candidate replaces the incumbent only for a fixed selection '
                        'score gain greater than numerical tolerance 1e-12. Ties retain the incumbent.')),
            uncertainty_rule=(('The incumbent rule above IS the one-standard-error rule of this run: the '
                               'incumbent score carries a subject-bootstrap standard error (reported in the '
                               'executed ranking when computed) and a challenger must clear it by that margin. '
                               'No confidence intervals beyond that are computed.')
                              if float(getattr(args, 'promotion_se', 0.0) or 0.0) > 0 else
                              ('No one-standard-error selection rule is used by this runner. '
                               'No standard errors or confidence intervals have been computed.')),
            revision_capabilities=['recipe_space', 'clf_space', 'exclude_source'],
            exclusion_effect='REVISE: exclude_source - <name> removes that source from the space AND '
                             'discards every candidate already scored through it, the incumbent '
                             'included, then reselects among what remains. Prefer it over recipe_space '
                             'when the objection is to one source: re-proposing a space cannot unseat '
                             'an incumbent that was itself fitted with the source you object to.',
            unsupported_revisions='The metric, adapter implementation, feature extraction, '
                                  'labels and patient partition cannot be changed by a review '
                                  'revision in this run. Record a verified issue there as a '
                                  'limitation; editing recipe_space does not edit an adapter.',
            feature_equivalence='Identical arrays are measured redundancy on the current '
                                'venue, not proof that extraction or aggregation is faulty. '
                                'Temporal aggregation may be a no-op for already aggregated '
                                'single-row features. A proposed recipe must be supported '
                                'by the actual adapter, not require an unimplemented transform.'),
        grouping_protocol=dict(
            selection=search.protocol,
            grouping='Keep patient subject identities and unique source-row identities fixed; '
                     'all rows of a patient stay together. Train/dev patients are disjoint '
                     'when official dev is present; otherwise use the fixed grouped OOF folds.',
            scoring='The current metric scores adapter rows; grouping does not imply patient pooling.'),
    )
    if hasattr(corpus, 'manifest'):
        constraints['backbone_protocol'] = dict(corpus.manifest)
    constraint_text = 'IMMUTABLE RUN CONSTRAINTS:\n' + json.dumps(constraints, indent=2)
    profile = (f'{corpus.CTX}: n_train={len(y)} / n_dev={len(first.get("dev_X", []))}; '
               f'p={first["train_X"].shape[1]}; feature_width_range={min(widths.values())}..'
               f'{max(widths.values())}; {len(set(first["train_subject"]))} subjects; '
               f'task={corpus.TASK}; selection={search.protocol}; metric={corpus.SEL_NAME}; ')
    profile += (f'target mean={float(y.mean()):.4f} sd={float(y.std()):.4f}' if kind == 'regression'
                else 'train class counts=' + json.dumps(dict(zip(*[
                    v.tolist() for v in np.unique(y, return_counts=True)]))))
    profile = constraint_text + '\n\nMEASURED REGIME:\n' + profile
    traj.log('regime_profile', dataset=corpus.CTX, profile=profile)

    mode = getattr(args, 'search_mode', 'joint')
    reference_records = []
    if mode == 'no_search' and references:
        print('  [no_search] %d carried-over reference candidate(s) kept for the regime profile '
              'but not scored' % len(references), flush=True)
    # Inherited reference pipelines are refitted first under this run's protocol and become the
    # starting incumbent.
    for index, candidate in (enumerate(references) if mode != 'no_search' else ()):
        before = len(search.records)
        search.evaluate([candidate['recipe']], [candidate['config']], round_no=-1,
                        on_record=lambda record: checkpoint('reference_search_running'))
        if len(search.records) != before + 1:
            raise RuntimeError('reference candidate was not evaluated exactly once')
        record = search.records[-1]
        reference_records.append({key: record[key] for key in
                                  ('id', 'valid', 'score', 'error') if key in record})
    if references and mode != 'no_search' and not any(record['valid'] for record in reference_records):
        checkpoint('reference_search_failed')
        raise RuntimeError('no supplied development reference candidate is valid in this run')

    seed_configs = []
    seed_path = Path(args.freeze_dir, ('reg' if kind == 'regression' else 'clf') +
                     '_space_' + corpus.CTX.replace('/', '_') + '.json')
    if seed_path.exists():
        seed_configs = json.loads(seed_path.read_text(encoding='utf-8')).get('configs', [])
    configs = _union(_union([item['config'] for item in references], seed_configs), corpus.propose_clf_space(
        proposer, profile, corpus.CTX, args.freeze_dir, traj, kind=kind))
    cg_build, cg_code = ((None, None) if kind == 'regression' else corpus.propose_codegen_method(
        proposer, profile, corpus.CTX, args.freeze_dir, traj))
    search.builder = lambda cfg, p: cg_build() if cfg == 'codegen' else corpus._make(cfg, p)
    all_configs = configs + (['codegen'] if cg_build is not None else [])
    if mode == 'fixed_recipe':
        fixed = _neutral_recipe(usable, references, 'fixed_recipe')
        print('  [fixed_recipe] recipe held at %s (first non-reference recipe of the derived '
              'space); searching %d model(s) only'
              % (json.dumps(fixed), len(all_configs)), flush=True)
        search.evaluate([fixed], all_configs,
                        on_record=lambda record: checkpoint('fixed_recipe_search_running'))
    elif mode == 'fixed_model':
        fixed_cfg = _neutral_probe(corpus, all_configs, references, 'fixed_model')
        print('  [fixed_model] model held at %s; searching %d recipe(s) only'
              % (json.dumps(fixed_cfg), len(usable)), flush=True)
        search.evaluate(usable, [fixed_cfg],
                        on_record=lambda record: checkpoint('fixed_model_search_running'))
    elif mode == 'no_search':
        fixed = _neutral_recipe(usable, references, 'no_search')
        fixed_cfg = _neutral_probe(corpus, all_configs, references, 'no_search')
        print('  [no_search] recipe held at %s and model held at %s; one candidate, no search'
              % (json.dumps(fixed), json.dumps(fixed_cfg)), flush=True)
        search.evaluate([fixed], [fixed_cfg],
                        on_record=lambda record: checkpoint('no_search_running'))
    elif mode == 'no_cpe':
        if search.best is None:
            fixed = _neutral_recipe(usable, references, 'no_cpe')
            fixed_cfg = _neutral_probe(corpus, all_configs, references, 'no_cpe')
            print('  [no_cpe] no exploration; starting the improver from %s x %s'
                  % (json.dumps(fixed), json.dumps(fixed_cfg)), flush=True)
            search.evaluate([fixed], [fixed_cfg],
                            on_record=lambda record: checkpoint('no_cpe_start_running'))
        else:
            print('  [no_cpe] no exploration; starting the improver from the carried-over reference '
                  '%s' % json.dumps(search.best['recipe']), flush=True)
    elif mode == 'staged':
        probe = _neutral_probe(corpus, all_configs, references, 'staged')
        print('  [staged] probing recipes with %s' % json.dumps(probe), flush=True)
        search.evaluate(usable, [probe],
                        on_record=lambda record: checkpoint('staged_recipe_search_running'))
        if search.best is None:
            checkpoint('search_failed')
            raise RuntimeError('staged recipe stage produced no valid candidate')
        staged_recipe = search.best['recipe']
        print('  [staged] recipe chosen with the probe model on %d recipes -> %s'
              % (len(usable), json.dumps(staged_recipe)), flush=True)
        search.evaluate([staged_recipe], all_configs,
                        on_record=lambda record: checkpoint('staged_model_search_running'))
    else:
        search.evaluate(usable, all_configs,
                        on_record=lambda record: checkpoint('initial_search_running'))

    held_recipe = (fixed if mode in ('fixed_recipe', 'no_search') else
                   staged_recipe if mode == 'staged' else None)
    held_config = fixed_cfg if mode in ('fixed_model', 'no_search') else None

    def arm_grid(recipes_now, configs_now):
        return ([held_recipe] if held_recipe is not None else recipes_now,
                [held_config] if held_config is not None else configs_now)

    if search.best is None:
        checkpoint('initial_search_failed')
        search.save(output)
        raise RuntimeError('Every complete recipe/model candidate failed; see development_summary.json')

    if memory_mode == 'evolving':
        memory = ResearchMemory(output, dict(corpus=corpus.CTX, task=corpus.TASK,
            metric=corpus.SEL_NAME, protocol=search.protocol, seed=args.seed,
            adapter_sha256=source_evidence.sha256 if source_evidence else None,
            initial_constraints=constraints))

    def finish_review(record):
        if memory is not None:
            record.setdefault('outcome', record['status']['state'])
            record['memory_update'] = memory.learn(
                proposer, record, search.records, incumbent=search.best, constraints=constraints,
                diagnostics=selection_diagnostics(search) if search.best else None, traj=traj)
        checkpoint('review_%d_complete' % record['round'])

    checkpoint('initial_search_complete_review_pending' if args.retrospective else
               'initial_search_complete_finalization_pending')

    for name, system in ROLE_SYSTEM.items():
        traj.add_system_prompt('development_' + name, system)
    traj.add_system_prompt('development_chair', CHAIR_SYSTEM)
    # 'front' gives the whole reserve to the first review; 'equal' lets round r use up to
    # reserve * r / R.
    if held_back:
        if reserve_schedule == 'front' or not args.retrospective:
            search.max_evaluations = total_budget
        if args.retrospective:
            print('  [budget] %d evaluations used; %d held back for what the review asks for (%s schedule)'
                  % (len(search.records), total_budget - len(search.records), reserve_schedule), flush=True)
    for round_no in range(1, args.retrospective + 1):
        try:
            from .progress import stage as _stage
        except ImportError:
            from progress import stage as _stage
        _stage('review round %d/%d' % (round_no, args.retrospective), traj)
        if held_back and reserve_schedule == 'equal':
            search.max_evaluations = args.max_evaluations + int(math.ceil(held_back * round_no / args.retrospective))
            print('  [budget] round %d may evaluate up to %d more (equal schedule)'
                  % (round_no, search.max_evaluations - len(search.records)), flush=True)
        selected = {k: v for k, v in search.best.items() if k != 'prediction'}
        measured = selection_diagnostics(search)
        incumbent_recipe_id = fingerprint(search.best['recipe'])
        incumbent_recipe_index = next(
            i for i, recipe in enumerate(usable) if fingerprint(recipe) == incumbent_recipe_id)
        incumbent_identity = dict(
            candidate_id=search.best['id'], recipe_index=incumbent_recipe_index,
            recipe=search.best['recipe'], config=search.best['config'],
            score=float(search.best['score']), venue=search.protocol,
            metric=corpus.SEL_NAME, n_query=measured['n_query'])
        authoritative_ranking = _authoritative_ranking(search, corpus.SEL_NAME)
        authoritative_prefix = ('CURRENT INCUMBENT (AUTHORITATIVE):\n' +
                                json.dumps(incumbent_identity, indent=2) +
                                '\n\nEXECUTED RANKING (AUTHORITATIVE):\n' +
                                json.dumps(authoritative_ranking, indent=2) +
                                '\nUse this executed ranking for every numeric tie, margin, '
                                'and next-candidate claim.')
        surface = explored_surface(search, usable, configs,
                                   search.max_evaluations - len(search.records))
        memory_context = memory.context() if memory is not None else None
        memory_prompt = ('\n\nACTIVE RESEARCH MEMORY:\n' + json.dumps(memory_context) +
                         '\n' + MEMORY_USE) if memory_context is not None else ''
        review_proposer = _ConstrainedReviewProposer(
            proposer, authoritative_prefix + '\n\n' + constraint_text +
            '\n\nCOMPUTED SELECTION DIAGNOSTICS:\n' +
            json.dumps(measured, indent=2) +
            '\n\nSEARCH COVERAGE SO FAR (what has been tried, and what has not):\n' +
            json.dumps(surface, indent=2) + memory_prompt)
        shared = dict(authoritative_ranking=authoritative_ranking,
                      immutable_constraints=constraints,
                      measured_diagnostics=measured,
                      search_coverage=surface,
                      corpus=corpus.CTX, task=corpus.TASK, primary_metric=corpus.SEL_NAME,
                      venue=search.protocol, incumbent=selected,
                      remaining_evaluations=search.max_evaluations - len(search.records),
                      revision_policy='Add recipe/model candidates; only measured strict improvement '
                                      'on the identical venue replaces the incumbent. Metric is fixed. '
                                      'Because addition is the only move, a direction that turns out '
                                      'not to help costs evaluations and retains the incumbent - it '
                                      'cannot lower what this run reports.',
                      scope='Adaptive development selection evidence; no independent final estimate.')
        if memory_context is not None:
            shared['research_memory'] = memory_context
        evidence = {role: dict(shared) for role in ROLE_SYSTEM}
        evidence['planner'].update(recipes=usable, classifier_configs=configs)
        evidence['engineer'].update(adapter_report=str(adapter_reasoning)[-1500:],
                                    feature_widths=widths, load_errors=search.load_errors)
        if source_evidence:
            evidence['engineer']['adapter_source_sha256'] = source_evidence.sha256
        evidence['reviewer'].update(candidates=search.summary()['candidates'],
                                   prediction_counts=dict(zip(*[v.tolist() for v in np.unique(
                                       search.best['prediction'], return_counts=True)])))
        tools = corpus._review_tools(search.view, usable, report)
        if source_evidence:
            tools['engineer']['adapter_source'] = (
                'literal keyword - inspect bounded lines of executed adapter source; '
                'use to check source/aggregation semantics, then compare with measured outputs',
                source_evidence)
        def candidate_scores(arg):
            return cached_recipe_evidence(search, usable, int(arg), corpus.SEL_NAME)
        tools['reviewer']['score_recipe'] = ('index - identified cached complete scores for this '
                                             'recipe and the current incumbent; no fitting',
                                             candidate_scores)
        if getattr(args, 'review_mode', 'roles') == 'random':
            drawn = random_revision(usable, configs, args.seed, round_no)
            if drawn is None:
                verdict, minutes = None, 'RANDOM REVISION: neither space can grow; nothing drawn.'
                status = dict(state='random_space_exhausted', complete=True, failed_roles=[],
                              chair_failed=False, chair_retried=False, review_mode='random')
            else:
                verdict = dict(step=drawn[0], note='random revision (no reasoning)', items=drawn[1])
                minutes = ('RANDOM REVISION: widen %s with %d item(s) drawn uniformly, seed %s, '
                           'round %d:\n%s' % (drawn[0], len(drawn[1]), args.seed, round_no,
                                               json.dumps(drawn[1], indent=1)))
                status = dict(state='random_revision', complete=True, failed_roles=[],
                              chair_failed=False, chair_retried=False, review_mode='random')
            traj.log('random_revision', dataset=corpus.CTX, choice=verdict, observation=status)
        else:
            left = total_budget - len(search.records)
            demand = (('review round %d of %d; %d evaluation(s) of the revision budget remain in this '
                       'run, %d of them available this round and the rest carried to later rounds'
                       % (round_no, args.retrospective, left, search.max_evaluations - len(search.records)))
                      if accept_policy == 'require_revision' and left > 0 else None)
            verdict, minutes, status = hold_review(
                review_proposer, evidence, tools=tools, prior_rounds=history or None,
                budget=args.review_checks, traj=traj, ctx=corpus.CTX, return_status=True,
                mode=getattr(args, 'review_mode', 'roles'), require_revision=demand)
        statuses.append(status)
        write_minutes(str(output), corpus.CTX, corpus.TASK, round_no, minutes)
        record = dict(round=round_no, status=status, decision=verdict,
                      incumbent=search.best['id'])
        if memory_context is not None:
            record['memory_version_used'] = memory_context['version']
        history.append(record)
        if verdict is None:
            if (accept_policy == 'require_revision' and getattr(args, 'review_mode', 'roles') != 'random'
                    and round_no < args.retrospective and total_budget - len(search.records) > 0):
                record['outcome'] = 'no_revision_this_round'
                traj.log('development_revision', observation=record)
                finish_review(record)
                continue
            finish_review(record)
            break
        if verdict['step'] == 'exclude_source':
            match = re.match(r"\s*['\"`]?([A-Za-z0-9_]+)", verdict.get('note') or '')
            name = match.group(1) if match else ''
            present = sorted({k for r in usable for k, v in r.items() if k != 'agg' and v})
            restrict = getattr(corpus, '_restrict_recipe_space', None)
            if name not in present or restrict is None:
                record['outcome'] = ('exclude_source_not_in_space' if restrict is not None
                                     else 'exclude_source_unsupported_by_corpus')
                record['comparison'] = dict(requested=name, present=present)
                traj.log('development_revision', observation=record)
                finish_review(record)
                continue
            previous = search.best['id'] if search.best else None
            try:
                remaining = restrict(usable, name)
            except ValueError as exc:
                record['outcome'] = 'exclude_source_empties_space'
                record['error'] = str(exc)
                traj.log('development_revision', observation=record)
                finish_review(record)
                break
            dropped, had_incumbent = search.invalidate_source(name)
            usable = remaining
            args.exclude_sources = ','.join(t for t in [getattr(args, 'exclude_sources', '') or '', name] if t)
            exclusions = sorted({n.strip() for n in args.exclude_sources.split(',') if n.strip()})
            constraints['excluded_sources'] = exclusions
            print('  [review] exclude_source %r: discarded %d scored candidate(s)%s; %d recipe(s) remain'
                  % (name, len(dropped), ' including the incumbent' if had_incumbent else '', len(usable)),
                  flush=True)
            _recipes, _configs = arm_grid(usable, configs + (['codegen'] if cg_build else []))
            result = search.evaluate(_recipes, _configs, round_no,
                                     on_record=lambda candidate, n=round_no:
                                         checkpoint('revision_%d_search_running' % n))
            if search.best is None:
                record['outcome'] = 'exclude_source_left_no_valid_candidate'
                traj.log('development_revision', observation=record)
                finish_review(record)
                raise RuntimeError('excluding %r left no valid candidate; the run cannot report a '
                                   'selection built on a disqualified source' % name)
            selected = {k: v for k, v in search.best.items() if k != 'prediction'}
            record['comparison'] = dict(result or {}, excluded=name, dropped=len(dropped),
                                        incumbent_disqualified=had_incumbent, previous_incumbent=previous)
            record['outcome'] = 'source_excluded'
            traj.log('development_revision', observation=record)
            finish_review(record)
            continue
        if verdict['step'] not in ('clf_space', 'recipe_space'):
            record['outcome'] = 'unsupported_revision_incumbent_retained'
            traj.log('development_revision', observation=record)
            finish_review(record)
            continue
        if ((verdict['step'] == 'recipe_space' and held_recipe is not None)
                or (verdict['step'] == 'clf_space' and held_config is not None)):
            record['outcome'] = 'revision_targets_held_axis_incumbent_retained'
            record['held_axis'] = verdict['step']
            record['search_mode'] = mode
            print('  [review] revision asks for %s, which %r holds fixed; incumbent retained'
                  % (verdict['step'], mode), flush=True)
            traj.log('development_revision', observation=record)
            finish_review(record)
            continue
        if len(search.records) >= search.max_evaluations:
            record['outcome'] = 'budget_exhausted_incumbent_retained'
            finish_review(record)
            break
        revision_dir = output / ('revision_%d' % round_no)
        revision_dir.mkdir(exist_ok=False)
        feedback = (constraint_text + '\n\nReview request (subject to the constraints above):\n' +
                    verdict['note'] + '\nFixed metric: ' + corpus.SEL_NAME +
                    '\nCurrent validated candidate: ' + json.dumps(selected) +
                    '\nPropose a bounded additive change. Do not change labels or split membership.' +
                    memory_prompt)
        revision_start_count = len(search.records)
        try:
            if verdict.get('items') is not None:
                proposed = list(verdict['items'])
                if verdict['step'] == 'clf_space':
                    configs = _union(configs, proposed)
                else:
                    usable = _union(usable, proposed)
            elif verdict['step'] == 'clf_space':
                proposed = corpus.propose_clf_space(proposer, profile + '\n' + feedback,
                    corpus.CTX, str(revision_dir), traj, kind=kind)
                configs = _union(configs, proposed)
            else:
                proposed = corpus.propose_recipe_space(proposer, report, extractors,
                    str(revision_dir), traj, feedback=feedback)
                proposed = corpus._restrict_recipe_space(proposed, args.exclude_sources)
                usable = _union(usable, proposed)
            if mode == 'no_cpe':
                _recipes, _configs = (([search.best['recipe']], proposed)
                                      if verdict['step'] == 'clf_space' else
                                      (proposed, [search.best['config']]))
                record['evaluation_grid'] = 'additions_x_incumbent'
            else:
                _recipes, _configs = arm_grid(usable, configs + (['codegen'] if cg_build else []))
            result = search.evaluate(
                _recipes, _configs, round_no,
                on_record=lambda candidate, n=round_no:
                    checkpoint('revision_%d_search_running' % n))
            record['comparison'] = result
            record['outcome'] = 'promoted' if result['promoted'] else 'incumbent_retained'
        except CheckpointPersistenceError:
            raise
        except Exception as exc:
            selected_id = search.best['id'] if search.best else None
            promoted = selected_id != record['incumbent'] and selected_id is not None
            record['outcome'] = ('revision_failed_incumbent_promoted' if promoted else
                                 'revision_failed_incumbent_retained')
            record['comparison'] = dict(
                round=round_no, incumbent=record['incumbent'], selected=selected_id,
                promoted=promoted, incomplete=True,
                new_evaluations=len(search.records) - revision_start_count)
            record['error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        traj.log('development_revision', observation=record)
        finish_review(record)

    gate = apply_reliability_gate(search, args)

    summary = search.save(output)
    summary['primary_metric'] = corpus.SEL_NAME
    summary['reliability_gate'] = gate
    if reference_source is not None:
        summary['reference_candidates'] = dict(reference_source,
                                                reevaluated=reference_records,
                                                status=('reference_kept_unscored_no_search'
                                                        if mode == 'no_search' else
                                                        'valid_reference_established'))
    if source_evidence:
        summary['adapter_source_sha256'] = source_evidence.sha256
    summary['immutable_constraints'] = constraints
    summary['measured_diagnostics'] = selection_diagnostics(search)
    frozen_proposals = sum(1 for step in (getattr(traj, 'trajectory', None) or [])
                           if str(step.get('type', '')).endswith('_frozen'))
    summary['agent'] = dict(backbone=getattr(corpus, 'backbone_name', 'run_corpus'),
                            llm_calls=traj.info['model_stats']['api_calls'],
                            frozen_agent_proposals=frozen_proposals,
                            agent_sourced_decisions=(traj.info['model_stats']['api_calls']
                                                     + frozen_proposals),
                            completed_reviews=sum(s['complete'] for s in statuses),
                            review_statuses=statuses, revisions=history,
                            research_memory=memory_status(),
                            revision_reserve=dict(fraction=reserve, held_back=held_back,
                                                  schedule=reserve_schedule),
                            review_accept_policy=accept_policy,
                            executed_codegen=cg_build is not None)
    if hasattr(load, 'cache_stats'):
        summary['adapter_cache'] = load.cache_stats
    raw_aggregate_stats = getattr(load, 'aggregate_cache_stats', None)
    if raw_aggregate_stats is not None:
        summary['raw_aggregation_cache'] = raw_aggregate_stats
    (output / 'development_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'recipes_executed.json').write_text(json.dumps(usable, indent=2), encoding='utf-8')
    (output / 'model_space_executed.json').write_text(json.dumps(configs, indent=2), encoding='utf-8')
    if cg_code:
        (output / 'codegen_model.py').write_text(cg_code, encoding='utf-8')
    traj.log('development_result', dataset=corpus.CTX, task=corpus.TASK, observation=summary)
    traj.save(str(output / 'agent_trajectory.json'))
    no_decisions = summary['agent']['agent_sourced_decisions'] == 0
    no_review = bool(args.retrospective) and not summary['agent']['completed_reviews']
    if args.require_llm and (no_decisions or no_review):
        checkpoint('finalization_blocked')
        reasons = []
        if no_decisions:
            reasons.append('no decision came from the agent: %d LLM call(s) and %d reused frozen '
                           'proposal(s)' % (summary['agent']['llm_calls'],
                                            summary['agent']['frozen_agent_proposals']))
        if no_review:
            reasons.append('the retrospective review did not complete: %d of %d round(s) finished '
                           '(%s)' % (summary['agent']['completed_reviews'], len(statuses),
                                     '; '.join(
                                         'round %d %s%s' % (
                                             i + 1, s.get('state', '?'),
                                             '' if not s.get('failed_roles')
                                             else ' failed=%s' % ','.join(s['failed_roles']))
                                         for i, s in enumerate(statuses)) or 'no round ran'))
        raise RuntimeError('Full Agent requires successful LLM execution/review: '
                           + '; and '.join(reasons)
                           + '. Artifacts saved, but this must not be reported as a completed '
                             'Agent experiment')
    checkpoint('complete', run_status='complete')
    print('DEVELOPMENT_RESULT ' + json.dumps(dict(metric=corpus.SEL_NAME, score=search.best['score'],
          recipe=search.best['recipe'], model=search.best['config'], agent=summary['agent'])), flush=True)
    return summary
