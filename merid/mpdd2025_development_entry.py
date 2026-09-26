"""Runs one MPDD-2025 context through the development loop on training data alone."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import shutil

import numpy as np

try:
    from .mpdd2025_corpus_facade import MPDDCorpusFacade
    from .adapter_agent import AdapterAgent, inspect_dataset, ADAPTER_SYSTEM
    from .development_agent import run_development
    from .proposer import ClaudeProposer
    from .trajectory import Trajectory
    from .registry import select_method
except ImportError:
    from mpdd2025_corpus_facade import MPDDCorpusFacade
    from adapter_agent import AdapterAgent, inspect_dataset, ADAPTER_SYSTEM
    from development_agent import run_development
    from proposer import ClaudeProposer
    from trajectory import Trajectory
    from registry import select_method


LABEL_FIELDS = {'binary': 'bin_category', 'ternary': 'tri_category', 'quinary': 'pen_category'}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path, parent):
    return path == parent or parent in path.parents


def _manifest_rows_contract(rows, ids, patients):
    if not isinstance(rows, list) or not rows:
        raise ValueError('training manifest must be a nonempty row list')
    observed_ids, observed_subjects = [], []
    for row in rows:
        identities = []
        if not isinstance(row, dict):
            raise ValueError('training manifest rows must be objects')
        for field, modality, suffix in (('audio_feature_path', 'A', '_audio_features'),
                                        ('video_feature_path', 'V', '_video_features')):
            value = row.get(field)
            if not value:
                continue
            if not isinstance(value, str) or '/' in value or '\\' in value:
                raise ValueError('manifest feature identity must be a basename')
            stem = Path(value).stem
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
            match = re.fullmatch(r'([0-9]+)(?:_([AV]))?_([0-9]+)', stem)
            if not match or (match.group(2) is not None and match.group(2) != modality):
                raise ValueError('invalid manifest segment identity')
            identities.append((match.group(1), match.group(1) + '_' + match.group(3)))
        if not identities or len(set(identities)) != 1:
            raise ValueError('manifest modalities disagree on segment identity')
        observed_subjects.append(identities[0][0])
        observed_ids.append(identities[0][1])
    if (len(set(observed_ids)) != len(observed_ids) or ids != observed_ids or
            patients != sorted(set(observed_subjects))):
        raise ValueError('stage patient/segment contract differs from original manifest order')
    return observed_subjects


def validate_stage(data_root, freeze_dir, context, task):
    MPDDCorpusFacade(context, task)
    data, frozen = Path(data_root).resolve(), Path(freeze_dir).resolve()
    stage = data.parent
    if data.name != 'data' or frozen != stage / 'frozen':
        raise ValueError('use the matching data/frozen directories of a prepared stage')
    provenance_path = frozen / 'staging_provenance.json'
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    if provenance.get('context') != context:
        raise ValueError('stage context does not match the requested context')
    adapter = frozen / ('adapter_' + context.replace('/', '_') + '.py')
    if (not adapter.is_file() or adapter.is_symlink() or
            _sha(adapter) != provenance.get('staged_adapter_sha256')):
        raise ValueError('staged adapter hash changed or file is missing')
    if any(p.is_symlink() or not p.is_file() for p in frozen.iterdir()):
        raise ValueError('frozen stage must contain regular files only')
    expected_frozen = {adapter.name, provenance_path.name}
    for source, digest in provenance.get('input_sha256', {}).items():
        name = Path(source).name
        if name in (adapter.name, 'Training_Validation_files.json'):
            continue
        if name in expected_frozen:
            raise ValueError('duplicate frozen artifact provenance')
        artifact = frozen / name
        if not artifact.is_file() or _sha(artifact) != digest:
            raise ValueError('copied frozen artifact hash changed or file is missing')
        expected_frozen.add(name)
    if {p.name for p in frozen.iterdir()} != expected_frozen:
        raise ValueError('frozen files differ from declared artifacts')
    if {p.name for p in data.iterdir()} != {'training', 'empty_test'}:
        raise ValueError('stage contains unexpected data roots')
    train, empty = data / 'training', data / 'empty_test'
    if train.is_symlink() or empty.is_symlink():
        raise ValueError('stage roots must not be aliases')
    shim_name = ('MM2025_Track2_Young.json' if context.startswith('MPDD-Young/')
                 else 'MM2025_Track1_Elderly.json')
    shim = empty / shim_name
    if ({p.name for p in empty.iterdir()} != {shim_name} or shim.is_symlink() or
            json.loads(shim.read_text(encoding='utf-8')) != []):
        raise ValueError('final-input shim must contain only an empty manifest')
    manifest = train / 'labels' / 'Training_Validation_files.json'
    target_manifest = manifest.resolve()
    training_source = target_manifest.parent.parent
    if (training_source.name != 'Training' or
            training_source.parent.name != context.split('/')[0] or
            provenance.get('input_sha256', {}).get(str(target_manifest)) != _sha(manifest)):
        raise ValueError('training manifest hash or source root changed')
    links = provenance.get('links')
    if not isinstance(links, list) or not links:
        raise ValueError('stage has no declared training links')
    expected = set()
    for item in links:
        relative = Path(item['link'])
        link = stage / relative
        if relative.is_absolute() or '..' in relative.parts or not _inside(link, train):
            raise ValueError('declared link escapes staged training root')
        target = Path(item['target']).resolve()
        if (not _inside(target, training_source) or not link.is_symlink() or
                link.resolve() != target or not target.is_file() or str(relative) in expected):
            raise ValueError('training link target changed, duplicated or is unavailable')
        expected.add(str(relative))
    actual = set()
    for path in train.rglob('*'):
        if path.is_dir():
            if path.is_symlink():
                raise ValueError('directory aliases are not allowed in staged data')
        else:
            actual.add(str(path.relative_to(stage)))
    if actual != expected:
        raise ValueError('stage files differ from the declared training links')
    ids = provenance.get('manifest_segment_ids', [])
    patients = provenance.get('allowed_patients', [])
    rows = json.loads(manifest.read_text(encoding='utf-8'))
    subjects = _manifest_rows_contract(rows, ids, patients)
    labels = np.asarray([row[LABEL_FIELDS[task]] for row in rows], dtype=float)
    n_classes = {'binary': 2, 'ternary': 3, 'quinary': 5}[task]
    if (labels.shape != (len(ids),) or not np.isfinite(labels).all() or
            np.any(labels != np.floor(labels)) or np.any(labels < 0) or np.any(labels >= n_classes)):
        raise ValueError('invalid declared training labels')
    return dict(context=context, adapter=str(adapter), data_root=str(data),
                freeze_dir=str(frozen), provenance_sha256=_sha(provenance_path),
                adapter_sha256=_sha(adapter), manifest_sha256=_sha(manifest),
                ids=ids, subjects=subjects,
                labels=labels.astype(int).tolist(), n_patients=len(patients), n_rows=len(ids))


def check_stage_rows(data, stage, task):
    if any(str(key).startswith(('dev_', 'test_')) for key in data):
        raise ValueError('MPDD development adapter must expose training rows only')
    for key, expected in [('train_ids', stage['ids']), ('train_subject', stage['subjects'])]:
        got = np.asarray(data.get(key))
        if got.shape != (len(expected),) or got.astype(str).tolist() != expected:
            raise ValueError(key + ' differs from the staged manifest')
    features = np.asarray(data.get('train_X'))
    if (features.ndim != 2 or features.shape[0] != stage['n_rows'] or features.shape[1] == 0
            or not np.issubdtype(features.dtype, np.number) or not np.isfinite(features).all()):
        raise ValueError('adapter features must be finite and match all staged rows')
    labels = np.asarray(data.get('train_' + task))
    if labels.shape != (stage['n_rows'],) or not np.array_equal(labels, stage['labels']):
        raise ValueError('adapter target differs from the original staged training labels')


def run(args):
    facade = MPDDCorpusFacade(args.ctx, args.task, args.selection_metric,
                              tuple(args.decoder_strategies.split(',')))
    facade.backbone_name = 'run_method:mpdd2025_development'
    stage = validate_stage(args.data_root, args.freeze_dir, args.ctx, args.task)
    output, seed = Path(args.output_dir).resolve(), Path(args.freeze_dir).resolve()
    data_root = Path(args.data_root).resolve()
    if (output.exists() or _inside(output, seed.parent) or _inside(seed.parent, output) or
            _inside(output, Path(__file__).resolve().parent)):
        raise ValueError('output must be fresh and outside staged inputs and executed source')
    output.mkdir(parents=True)
    shutil.copytree(seed, output / 'frozen')
    args.freeze_dir, args.output_dir = str(output / 'frozen'), str(output)
    safe = args.ctx.replace('/', '_')
    traj = Trajectory(model=args.llm_model,
                      config=dict(corpus=args.ctx, task=args.task, evaluation='development'),
                      stream_path=str(output / 'agent_trajectory.jsonl'))
    try:
        proposer = ClaudeProposer(model=args.llm_model, provider=args.proposer,
                                  base_url=args.llm_base_url, api_key=args.llm_api_key)
        proposer.traj = traj
        traj.add_system_prompt('adapter_codegen', ADAPTER_SYSTEM)
        (output / 'development_input_manifest.json').write_text(json.dumps(stage, indent=2), encoding='utf-8')
        report = inspect_dataset({'train_root': str(data_root / 'training')})
        profile = (f'{args.ctx}: {stage["n_rows"]} segments / {stage["n_patients"]} subjects; '
                   f'task={args.task}; grouped training-only OOF; metric={args.selection_metric}. '
                   'The implemented route searches the original classical model/head space. '
                   'Deep/CREATE/competition-framework selections cannot execute in this route.')
        available = ['sklearn_svc', 'sklearn_logreg', 'sklearn_gb', 'sklearn_rf_et']
        method, reasoning = select_method(profile + '\n' + report[:1500], proposer,
                                           available_methods=available)
        route = dict(selected_method=method, selection_reasoning=reasoning,
                     executed_route=None, protocol=facade.manifest,
                     executable_methods=available)
        traj.log('method_select', prompt=profile, reasoning=reasoning, choice=method, dataset=args.ctx)
        if method not in ('sklearn_svc', 'sklearn_logreg', 'sklearn_gb', 'sklearn_rf_et'):
            route['status'] = 'selected_route_unimplemented'
            (output / 'method_route.json').write_text(json.dumps(route, indent=2), encoding='utf-8')
            traj.save(str(output / 'agent_trajectory.json'))
            raise RuntimeError('selected method has no MPDD development runner: ' + str(method))
        route.update(status='classical_search_selected', executed_route='original classical candidate space',
                     selected_head_is_advisory=True)
        (output / 'method_route.json').write_text(json.dumps(route, indent=2), encoding='utf-8')
        traj.log('benchmark_metric_fixed', dataset=args.ctx, task=args.task, choice=args.selection_metric)
        recipes = facade.propose_recipe_space(proposer, report, [], args.freeze_dir, traj)
        recipes = facade._restrict_recipe_space(recipes, args.exclude_sources)
        copied_adapter = Path(args.freeze_dir) / Path(stage['adapter']).name
        if _sha(copied_adapter) != stage['adapter_sha256']:
            raise ValueError('copied adapter hash changed before execution')
        raw_load, code, _, reasoning = AdapterAgent(proposer).create_adapter(
            {'train_root': str(data_root / 'training')},
            f'MPDD-2025 {args.ctx} {args.task}; use only the verified training stage.',
            max_iters=0, freeze_path=str(copied_adapter), data_scope='development',
            adapter_cache_bytes=256 * 1024 * 1024, require_row_ids=True)
        if raw_load is None or _sha(copied_adapter) != stage['adapter_sha256']:
            raise RuntimeError('verified frozen MPDD adapter failed; preserve and diagnose this stage')

        def load(recipe=None):
            data = raw_load(recipe)
            check_stage_rows(data, stage, args.task)
            return data

        traj.log('adapter_codegen', dataset=args.ctx, ok=True, code=code, reasoning=reasoning)
        summary = run_development(facade, args, proposer, load, recipes, report, [], reasoning,
                                  traj, adapter_source=code)
        summary['method_route'] = route
        summary['stage_provenance_sha256'] = stage['provenance_sha256']
        summary['adapter_cache'] = raw_load.cache_stats
        (output / 'development_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        traj.log('mpdd_development_route', observation=route)
        traj.save(str(output / 'agent_trajectory.json'))
        return summary
    finally:
        traj.close()
