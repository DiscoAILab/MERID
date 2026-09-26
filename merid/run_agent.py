"""MPDD-2026 entry point: head search and review on the development folds."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import Harness
from manager import optimize
from profiler import profile as build_profile
from proposer import ClaudeProposer, RandomProposer
try:
    from trajectory import Trajectory
except ImportError:
    Trajectory = None
from mpdd2026_metrics import METRIC_CONTRACT
from mpdd2026_joint_search import (HeadProposer, validate_envelope,
                                 _validate_result, _joint)


class RequiredLLMFailure(BaseException):
    pass


class _RequiredLLMProposer:
    def __init__(self, proposer):
        self.proposer = proposer
        self.successful_proposals = 0

    @property
    def last_reasoning(self):
        return getattr(self.proposer, "last_reasoning", "")

    @property
    def last_attempts(self):
        return getattr(self.proposer, 'last_attempts', [])

    @property
    def total_api_calls(self):
        return getattr(self.proposer, 'total_api_calls', None)

    def complete(self, system, prompt, **kwargs):
        return self.proposer.complete(system, prompt, **kwargs)

    def propose(self, profile, history):
        try:
            result = self.proposer.propose(profile, history)
            self.successful_proposals += 1
            return result
        except Exception as exc:
            raise RequiredLLMFailure("required LLM proposal failed: %s" %
                                     type(exc).__name__) from None


def build_proposer(args):
    if args.proposer == "random":
        return RandomProposer(seed=args.seed)
    strict_config = (getattr(args, "evaluation", "all") == "development"
                     and getattr(args, "require_llm", False))
    return ClaudeProposer(model=args.llm_model or "claude-opus-4-8",
                          provider="anthropic", api_key=args.llm_api_key,
                          base_url=args.llm_base_url, strict_config=strict_config)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _snapshot_code(run_dir):
    project = Path(__file__).resolve().parents[1]
    relative_paths = (
        "merid/run_agent.py", "merid/evaluate.py", "merid/manager.py",
        "merid/proposer.py", "merid/profiler.py", "merid/space.py",
        "merid/mpdd2026_protocol.py",
        "merid/mpdd2026_metrics.py",
        "merid/mpdd2026_joint_search.py",
        "merid/mpdd2026_data.py",
        "merid/mpdd2026_heads.py",
    )
    hashes = {}
    for relative in relative_paths:
        source = project / relative
        destination = Path(run_dir, "code_snapshot", relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[relative] = _sha256(destination)
    return hashes


def _development_inputs(args):
    train_root = Path(args.train_root).resolve()
    personality = Path(args.personality_npy).resolve()
    split_family = Path(args.split_root).resolve() / args.fold_task
    entries = []
    for path in (train_root / "split_labels_train.csv", personality):
        entries.append({"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)})
    fold_files = [split_family / ("fold%d.csv" % number) for number in range(args.n_folds)]
    entries.append({
        "path": str(split_family),
        "files": [{"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
                  for path in fold_files],
    })
    return entries


def _joint_reserve_totals(best, budget, fit_limit, rounds, reserve):
    # Accepted revisions extend both budgets by the reserve; the totals are recomputed and
    # matched against the review record.
    rounds = max(0, int(rounds))
    expect_candidates = int(math.ceil(budget * max(0.0, float(reserve)))) if rounds else 0
    expect_fits = int(math.ceil(fit_limit * max(0.0, float(reserve)))) if rounds else 0
    review = best.get('review') if isinstance(best, dict) else None
    executed = review.get('revisions_executed') if isinstance(review, dict) else None
    if (not isinstance(review, dict) or type(executed) is not int
            or not 0 <= executed <= rounds
            or review.get('reserve_candidates') != expect_candidates
            or review.get('reserve_fits') != expect_fits):
        raise RuntimeError('joint review accounting does not match the declared revision reserve')
    return budget + executed * expect_candidates, fit_limit + executed * expect_fits


def _validate_joint_completion(history, best, harness, budget, fit_limit, rounds=0, reserve=0.0):
    if not isinstance(best, dict):
        raise RuntimeError('joint search did not produce complete components')
    budget_total, fit_total = _joint_reserve_totals(best, budget, fit_limit, rounds, reserve)
    if not 2 <= len(history) <= budget_total:
        raise RuntimeError('joint search did not produce complete components')
    evidence, costs = best.get('evaluation', {}), best.get('fit_budget', {})
    ids = best.get('component_ids', {})
    families = {'binary_phq', 'hierarchical_ternary'}
    actual = [r.get('fit_calls') for r in history]
    if (set(ids) != families or set(best.get('choice', {})) != families
            or any(type(value) is not int or value < 0 for value in actual)
            or costs.get('limit') != fit_total or costs.get('used') != sum(actual)
            or not 0 <= costs.get('used', -1) <= fit_total):
        raise RuntimeError('joint component identities or fit accounting are incomplete')
    stop = costs.get('stop_reason')
    if stop not in {'candidate_budget_complete', 'insufficient_remaining_fit_budget',
                    'candidate_space_exhausted'} or (stop == 'candidate_budget_complete'
                                                     and len(history) != budget_total):
        raise RuntimeError('joint search stopped without a valid budget termination')
    components = {}
    for family, identifier in ids.items():
        matches = [r for r in history if r.get('id') == identifier and r.get('head_family') == family]
        if (len(matches) != 1 or matches[0].get('valid') is not True
                or matches[0].get('choice') != best['choice'][family]):
            raise RuntimeError('joint best does not resolve to valid component history')
        components[family] = matches[0]
        _validate_result(matches[0].get('evaluation'), family, harness)
    protocol = harness.fold_protocol
    if (evidence.get('phq_metric_contract') != METRIC_CONTRACT
            or evidence.get('protocol_fingerprint') != protocol.protocol_fingerprint
            or evidence.get('protocol_evidence_fingerprint') != protocol.evidence_fingerprint
            or evidence.get('patient_ids') != list(protocol.patient_ids)
            or evidence.get('nested_final_estimate') is not False
            or evidence.get('cache_only_recombination') is not True
            or evidence.get('component_ids') != ids):
        raise RuntimeError('joint best lacks complete shared raw-PHQ metric contract evidence')
    metrics = [float(evidence.get(key, float('nan'))) for key in ('binF1', 'terF1', 'ccc')]
    score = float(best.get('track', float('nan')))
    if (not all(math.isfinite(value) for value in metrics + [score])
            or abs(sum(metrics) / 3.0 - score) > 1e-12):
        raise RuntimeError('joint track does not match its component metrics')
    expected = _joint(components['binary_phq'], components['hierarchical_ternary'])
    if evidence != expected or abs(score - expected['track']) > 1e-12:
        raise RuntimeError('joint best metrics are not bound to their component evaluations')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", default="all", choices=["all", "development"])
    ap.add_argument("--track", default="Track2", choices=["Track1", "Track2"])
    ap.add_argument("--proposer", default="anthropic", choices=["anthropic", "random"])
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--search_mode", choices=["phq_only", "joint_heads"], default="phq_only")
    ap.add_argument("--search_axes", choices=["all", "fixed_recipe", "fixed_model"],
                    default="all")
    ap.add_argument("--fit_budget", type=int, default=720)
    ap.add_argument("--reference_components", help="Optional JSON of typed, freshly evaluated joint seed references")
    ap.add_argument("--retrospective", type=int, default=0,
                    help="review rounds held after the candidate budget is spent (joint_heads only). "
                         "The EGE review; each round may extend the search by the reserve.")
    ap.add_argument("--review_checks", type=int, default=2,
                    help="read-only checks each reviewer may run before reporting")
    ap.add_argument("--review_mode", choices=["roles", "single_agent"], default="roles",
                    help="roles = three participants and a chair; single_agent = one reviewer "
                         "holding all the evidence, for the ablation of role separation")
    ap.add_argument("--revision_reserve", type=float, default=0.25,
                    help="candidates and fits held back for what a review asks for, as a fraction "
                         "of the budget and ADDED to it")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm_model", default=None, help="model id (default claude-opus-4-8)")
    ap.add_argument("--llm_base_url", default=None, help="API base URL (default: ANTHROPIC_BASE_URL or the SDK's)")
    ap.add_argument("--llm_api_key", default=None, help="or set ANTHROPIC_API_KEY; do NOT commit")
    ap.add_argument("--output_dir", default="output")
    ap.add_argument("--exp_name", default=None)
    ap.add_argument("--train_root")
    ap.add_argument("--personality_npy")
    ap.add_argument("--split_root")
    ap.add_argument("--fold_task")
    ap.add_argument("--n_folds", type=int, default=10)
    ap.add_argument("--require_llm", action="store_true")
    args = ap.parse_args(argv)
    if not 1 <= args.budget <= 256:
        ap.error("--budget must be between 1 and 256")
    if not 1 <= args.n_folds <= 256:
        ap.error("--n_folds must be between 1 and 256")
    if args.fit_budget < 1:
        ap.error("--fit_budget must be positive")
    joint = args.search_mode == 'joint_heads'
    if joint and (args.evaluation != 'development' or args.budget < 2):
        ap.error("joint_heads requires development evaluation and candidate budget >= 2")
    if joint and args.fit_budget < 7 * args.n_folds:
        ap.error("joint_heads fit budget must cover both seed components (7 * n_folds)")
    references, reference_path, reference_bytes = None, None, None
    if args.reference_components:
        if not joint:
            ap.error('--reference_components requires joint_heads')
        reference_path = Path(args.reference_components).resolve()
        try:
            reference_bytes = reference_path.read_bytes()
            references = json.loads(reference_bytes.decode('utf-8-sig'))
            if (not isinstance(references, dict) or not references
                    or not set(references) <= {'binary_phq', 'hierarchical_ternary'}):
                raise ValueError('reference families must be a nonempty supported subset')
            references = {family: validate_envelope(value) for family, value in references.items()}
            if any(value['head_family'] != family for family, value in references.items()):
                raise ValueError('reference envelope head_family must match its mapping key')
        except (OSError, ValueError) as exc:
            ap.error('--reference_components must be an existing valid typed JSON: ' + str(exc))
    if args.evaluation == "development" and any(
            getattr(args, name) is None
            for name in ("train_root", "personality_npy", "split_root", "fold_task")):
        ap.error("development evaluation requires --train_root, --personality_npy, --split_root, and --fold_task")
    if args.require_llm:
        if args.proposer == "random":
            ap.error("--require_llm cannot be used with --proposer random")
        key_names = ("ANTHROPIC_API_KEY",)
        if not (args.llm_api_key or any(os.environ.get(name) for name in key_names)):
            ap.error("--require_llm needs an API key argument or provider environment variable")

    if args.exp_name is not None and not re.fullmatch(r'[A-Za-z0-9_-]+', args.exp_name):
        ap.error('--exp_name must be a simple name without path components')
    if args.evaluation == "development":
        output_root = Path(args.output_dir).resolve()
        protected_roots = [Path(args.train_root).resolve(), Path(args.split_root).resolve(),
                           Path(__file__).resolve().parent]
        if any(output_root == root or root in output_root.parents for root in protected_roots):
            ap.error('development output must be outside inputs and executed source')
        if reference_path and (output_root == reference_path or output_root in reference_path.parents):
            ap.error('development output must not contain the reference input')
    if args.evaluation == "development":
        harness = Harness(
            track=args.track, split_root=args.split_root, n_folds=args.n_folds,
            data_scope="development", train_root=args.train_root,
            personality_npy=args.personality_npy, fold_task=args.fold_task)
    else:
        harness = Harness(track=args.track)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    exp = args.exp_name or f"merid_{args.proposer}_{args.track}"
    run_dir = os.path.join(args.output_dir, f"{exp}_{stamp}")
    os.makedirs(run_dir, exist_ok=False)
    log_path = os.path.join(run_dir, "trajectory.jsonl")
    print(f"output dir = {run_dir} | proposer = {args.proposer} | budget = {args.budget}", flush=True)
    _atomic_json(Path(run_dir, "status.json"), {"status": "running"})
    try:
        profile = build_profile(harness)
        proposer = build_proposer(args)
        traj = (Trajectory(model=args.llm_model or '', config=dict(corpus='MPDD-AVG2026', track=args.track,
                                                                    evaluation=args.evaluation, seed=args.seed))
                if Trajectory is not None else None)
        if traj is not None:
            try:
                proposer.traj = traj
            except AttributeError:
                pass
        if joint and args.proposer != 'random':
            proposer = HeadProposer(proposer, audit_path=Path(run_dir, 'joint_proposals.jsonl'),
                                    search_axes=args.search_axes)
        if args.require_llm:
            proposer = _RequiredLLMProposer(proposer)
        provenance = {
            "evaluation": args.evaluation, "track": args.track, "seed": args.seed,
            "budget": args.budget, "proposer": args.proposer,
            "search_mode": args.search_mode,
            "fit_budget_limit": args.fit_budget if joint else None,
            "llm_model": None if args.proposer == "random" else args.llm_model or "claude-opus-4-8",
            "require_llm": args.require_llm,
            "source_snapshot_sha256": _snapshot_code(run_dir),
            "execution_source": "live source; snapshot files are provenance copies",
        }
        if args.evaluation == "development":
            provenance.update(
                protocol_fingerprint=harness.fold_protocol.protocol_fingerprint,
                protocol_evidence_fingerprint=harness.fold_protocol.evidence_fingerprint,
                patient_count=len(harness.fold_protocol.patient_ids),
                fold_task=args.fold_task, n_folds=args.n_folds,
                inputs=_development_inputs(args),
                phq_metric_contract=METRIC_CONTRACT,
            )
        if reference_path:
            Path(run_dir, 'reference_components.json').write_bytes(reference_bytes)
            provenance['reference_components'] = {'path': str(reference_path),
                'sha256': _sha256(Path(run_dir, 'reference_components.json')),
                'fresh_fit': True, 'candidate_budget_charged': True}
        _atomic_json(Path(run_dir, "provenance.json"), provenance)
        search_options = ({'search_mode': args.search_mode, 'fit_budget': args.fit_budget,
                           'retrospective': args.retrospective,
                           'review_checks': args.review_checks,
                           'review_mode': args.review_mode,
                           'revision_reserve': args.revision_reserve,
                           'search_axes': args.search_axes,
                           'ctx': 'MPDD-2026/' + str(args.track)}
                          if joint else {})
        if references is not None:
            search_options['reference_components'] = references
        history, best = optimize(harness, proposer, budget=args.budget, seed=args.seed,
                                 log_path=log_path, profile=profile, **search_options)
        if args.require_llm:
            score_key = 'component_score' if joint else 'track'
            llm_scored = [item for item in history if item.get("source") == "llm"
                          and (not joint or item.get('valid') is True)
                          and item.get(score_key) is not None
                          and math.isfinite(float(item[score_key]))]
            if not llm_scored or proposer.successful_proposals < 1:
                raise RuntimeError("required LLM run has no finite LLM-selected evaluation")
        valid_best = (best is not None and best.get("track") is not None
                      and math.isfinite(float(best["track"])))
        if args.evaluation == "development" and not joint:
            finite = [item for item in history if item.get("track") is not None
                      and math.isfinite(float(item["track"]))]
            if any(item.get("evaluation", {}).get("phq_metric_contract") != METRIC_CONTRACT
                   for item in finite):
                raise RuntimeError("development result lacks the versioned raw-PHQ metric contract")
        if joint:
            _validate_joint_completion(history, best, harness, args.budget, args.fit_budget,
                                       args.retrospective, args.revision_reserve)
        elif args.evaluation == "development" and (len(history) != args.budget or not valid_best):
            raise RuntimeError("development search did not complete its budget with a scored candidate")
        ranked = [best] if joint else sorted([h for h in history if h.get("track") is not None
                         and math.isfinite(float(h["track"]))],
                        key=lambda h: h["track"], reverse=True)
        print("\n==================== AGENT SEARCH RANKING (OOF track) ====================")
        for h in ranked:
            c = h["choice"]
            if joint:
                e = h['evaluation']
                print(f"{h['track']:.4f}  binF1={e['binF1']:.3f} terF1={e['terF1']:.3f} "
                      f"ccc={e['ccc']:.3f}  [joint_cached_recombination]")
            else:
                print(f"{h['track']:.4f}  binF1={h['binF1']:.3f} terF1={h['terF1']:.3f} ccc={h['ccc']:.3f}  "
                      f"{c['audio']}/{c['video']}/{c['mode']}/{c['model']}/k{c['k']}/s{c['shrink']}  [{h.get('source')}]")
        if best:
            print(f"\nBEST (best-found under budget {args.budget}): OOF track = {best['track']:.4f}\n  {best['choice']}")

        results = {
            "exp_name": exp, "track": args.track, "proposer": args.proposer, "budget": args.budget,
            "evaluation": args.evaluation,
            "search_mode": args.search_mode,
            "metric": "(binaryMacroF1 + ternaryMacroF1 + PHQ_CCC)/3, adaptive OOF",
            "agent_path": "proposer_manager",
            "required_llm_successful_proposals": getattr(proposer, "successful_proposals", None),
            "not_integrated": ["four_role_retrospective", "codegen"] + ([] if joint else ["ternary_grid"]),
            "profile": profile,
            "ranking": [dict(best, source='joint_cached_recombination')] if joint else [{"track": h["track"], "choice": h["choice"], "binF1": h["binF1"],
                         "terF1": h["terF1"], "ccc": h["ccc"], "source": h.get("source"),
                         "evaluation": h.get("evaluation")} for h in ranked],
            "best": dict(best, source='joint_cached_recombination') if joint else ({"choice": best["choice"], "track": best["track"],
                      "source": best.get("source"), "evaluation": best.get("evaluation")}
                     if best else None),
        }
        if joint:
            results.update(component_history=history, fit_budget=best['fit_budget'],
                           llm_api_calls=getattr(proposer, 'total_api_calls', 0),
                           llm_api_call_limit=2 * (args.budget - 2),
                           agent_path='proposer_manager_joint_heads')
        if args.evaluation == "development":
            results.update(
                evaluation_protocol="adaptive_oof_threshold_calibration",
                nested_final_estimate=False,
                protocol_fingerprint=harness.fold_protocol.protocol_fingerprint,
                phq_metric_contract=METRIC_CONTRACT,
                metric="(binaryMacroF1 + ternaryMacroF1 + raw PHQ CCC 0..27)/3, adaptive OOF",
            )
        if traj is not None:
            results['llm_model_stats'] = dict(traj.info['model_stats'])
            traj.save(os.path.join(run_dir, 'agent_trajectory.json'))
        _atomic_json(Path(run_dir, "results.json"), results)
        _atomic_json(Path(run_dir, "status.json"), {"status": "complete"})
        print(f"\nsaved -> {run_dir}/results.json  |  trajectory -> {log_path}")
        return run_dir
    except BaseException as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
        secrets = [args.llm_api_key, os.environ.get('ANTHROPIC_API_KEY')]
        sanitized = error
        for secret in secrets:
            if secret:
                sanitized = sanitized.replace(secret, '[redacted]')
        _atomic_json(Path(run_dir, "status.json"), {"status": "failed", "error": sanitized})
        if sanitized != error:
            raise RuntimeError(sanitized) from None
        raise


if __name__ == "__main__":
    main()
