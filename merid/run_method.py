"""MPDD-2025 entry point and the recipe, model and metric proposal steps shared with run_corpus."""
import argparse
import os
import re

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin

from merid.core.grounded_state_construction.adapter_agent import (
    ADAPTER_SYSTEM, AdapterAgent, inspect_dataset)
from merid.core.coupled_pipeline_exploration.operation import (
    OperationAgent, Registry, retrieve_grounding)
from merid.llm.proposer import ClaudeProposer
from merid.core.coupled_pipeline_exploration.registry import ensure_repo, select_method
from merid.benchmarks.mpdd2025.run_all import (
    CONFIGS, WINDOWS, agent_classify, grouped_folds, make_clf, oof_clf, official, spec_for)
from merid.utils.trajectory import Trajectory

DEFAULT_SPACE = [{"audio": "wav2vec", "video": v, "agg": g, "ind_emb": True, "meta": m}
                 for v in ("densenet", "resnet", "openface")
                 for g in ("meanstd", "meanstd_minmax")
                 for m in (False, True)]
PROBE_CFG = {"model": "svc_linear", "k": 128, "C": 0.1}

RECIPE_SPACE_SYSTEM = (
    "You DESIGN the candidate feature-recipe SPACE for a SMALL-SAMPLE (p>>n) multimodal depression task. "
    "Reason from the available extractors + their content stats and the grounding, then output ONLY a JSON list "
    "of candidate recipes. Each recipe is a dict: "
    '{"audio": "wav2vec"|"opensmile"|"mfccs"|null, "video": "densenet"|"resnet"|"openface"|null, '
    '"agg": "meanstd"|"meanstd_minmax"|"meanstd_p25p75"|"meanstd_delta", "ind_emb": bool, "meta": bool}. '
    "meta = structured personality/context (big5+family+disease, the 'P'); ind_emb = the individualEmbedding "
    "vector; null audio/video DROPS that modality; 'meanstd_delta' adds frame-to-frame dynamics. COVER "
    "modality-combination ablations (audio-only, video-only, personality-only, A+V, A+V+P) AND the extractor/"
    "aggregation choices worth testing — including at least one TEMPORAL-DYNAMICS ('meanstd_delta') candidate, "
    "since plain mean/std discards how the signal evolves. Include 8-14 DISTINCT recipes. An OOF search picks "
    "the winner — propose a good, diverse SPACE, not one guess. Output ONLY the JSON list."
)


def _norm_recipe(r):
    if not isinstance(r, dict):
        return None
    rec = {"audio": r.get("audio"), "video": r.get("video"), "agg": r.get("agg", "meanstd"),
           "ind_emb": bool(r.get("ind_emb", False)), "meta": bool(r.get("meta", False))}
    for k in ("audio", "video"):
        if rec[k] in ("", "none", "None", "null"):
            rec[k] = None
    if rec["agg"] not in ("meanstd", "meanstd_minmax", "meanstd_p25p75", "meanstd_delta"):
        rec["agg"] = "meanstd"
    if rec["audio"] not in ("wav2vec", "opensmile", "mfccs", None):
        rec["audio"] = None
    if rec["video"] not in ("densenet", "resnet", "openface", None):
        rec["video"] = None
    if rec["audio"] is None and rec["video"] is None and not rec["meta"] and not rec["ind_emb"]:
        return None
    return rec


def _parse_recipe_list(txt):
    import json as _json
    import re as _re
    out, seen = [], set()
    m = _re.search(r"\[.*\]", txt or "", _re.S)
    if m:
        try:
            for r in _json.loads(m.group(0)):
                rec = _norm_recipe(r)
                if rec is None:
                    continue
                k = _json.dumps(rec, sort_keys=True)
                if k not in seen:
                    seen.add(k); out.append(rec)
        except Exception:
            pass
    return out


def propose_recipe_space(proposer, report, ctx, freeze_dir, traj=None, grounding=""):
    import hashlib
    import json as _json
    safe = ctx.replace("/", "_").replace("\\", "_")
    fp = os.path.join(freeze_dir, f"recipe_space_{safe}.json")
    if os.path.exists(fp):
        obj = _json.load(open(fp))
        print(f"  [recipe-space] reused {len(obj['recipes'])} FROZEN recipes ({fp})", flush=True)
        if traj is not None:
            traj.log("recipe_space_frozen", dataset=ctx, choice=obj["recipes"], sha256=obj.get("sha256"))
        return obj["recipes"]
    from merid.llm.retrieval import decide_with_optional_search
    prompt = (f"Dataset: {ctx}\n\nOBSERVED inspection (available extractors, shapes, value statistics, "
              f"label fields) — this is measured from THIS data:\n{report[:2600]}\n\n"
              f"Design the candidate recipe list now.")
    txt = decide_with_optional_search(proposer, RECIPE_SPACE_SYSTEM, prompt, traj=traj, ctx=ctx,
                                      tag="recipe-space")
    recipes = _parse_recipe_list(txt)
    if not recipes:
        recipes = DEFAULT_SPACE
        print(f"  [recipe-space] proposal empty/unparsable -> DEFAULT_SPACE ({len(recipes)})", flush=True)
    os.makedirs(freeze_dir, exist_ok=True)
    payload = {"ctx": ctx, "recipes": recipes, "reasoning": txt}
    payload["sha256"] = hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with open(fp, "w") as fh:
        _json.dump(payload, fh, indent=2)
    print(f"  [recipe-space] agent proposed {len(recipes)} recipes -> FROZEN {fp}", flush=True)
    if traj is not None:
        traj.log("recipe_space_propose", dataset=ctx, reasoning=txt, choice=recipes, sha256=payload["sha256"])
    return recipes


RECIPE_CV_REPEATS = 3
RECIPE_STD_PENALTY = 0.5
RECIPE_MIN_COVERAGE = 0.95


def _recipe_task_score(X, groups, y):
    offs, covs = [], []
    for s in range(RECIPE_CV_REPEATS):
        oof, _, _ = oof_clf(X, X[:1], y, PROBE_CFG, grouped_folds(y, groups, seed=s))
        mk = oof >= 0
        cov = float(mk.mean())
        covs.append(cov)
        if cov < RECIPE_MIN_COVERAGE:
            return None
        offs.append(official(y[mk], oof[mk])[4])
    return (float(np.mean(offs) - RECIPE_STD_PENALTY * np.std(offs)),
            float(np.mean(offs)), float(np.std(offs)), float(np.mean(covs)))


def select_recipe(load, recipes, tasks, traj, ctx):
    score_tasks = list(dict.fromkeys(t for t, _ in tasks))
    best = None
    for rc in recipes:
        try:
            data = load(rc)
        except Exception as exc:
            print(f"  [recipe {rc}] load FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        X = np.asarray(data["train_X"], float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            print(f"  [recipe {rc}] skipped: train_X shape {X.shape}", flush=True)
            continue
        groups = data.get("train_subject")
        scored, detail, invalid = [], {}, False
        for t in score_tasks:
            y = np.asarray(data[f"train_{t}"], int)
            if len(set(y.tolist())) < 2:
                continue
            r = _recipe_task_score(X, groups, y)
            if r is None:
                invalid = True; detail[t] = "INVALID(low-coverage)"; break
            scored.append(r[0]); detail[t] = f"off={r[1]:.3f}±{r[2]:.3f} cov={r[3]:.2f}"
        p = X.shape[1]
        if invalid or not scored:
            print(f"  [recipe] {rc} -> p={p} INVALID {detail}", flush=True)
            if traj is not None:
                traj.log("recipe_select", dataset=ctx, choice=rc, p=int(p), valid=False, detail=detail)
            continue
        score = float(np.mean(scored))
        print(f"  [recipe] {rc} -> p={p} robust-official={score:.4f} {detail}", flush=True)
        if traj is not None:
            traj.log("recipe_select", dataset=ctx, choice=rc, p=int(p),
                     robust_official=round(score, 4), detail=detail)
        if best is None or score > best[0]:
            best = (score, rc, data)
    return best


CLF_SPACE_SYSTEM = (
    "You DESIGN the candidate CLASSIFIER space for a SMALL-SAMPLE (p>>n) multimodal depression task. "
    "Reason from the data regime and the grounding, then output ONLY a JSON list of candidate scikit-learn "
    'configs. Each config: {"model": "svc_linear"|"svc_rbf"|"logreg"|"gb"|"rf"|"et", '
    '"k": 32|64|128|256 (SelectKBest features), "C": 0.03..1.0 (svc/logreg), "lr": 0.03..0.1 (gb), '
    '"depth": 1..3 (gb)}. Span model FAMILIES and regularisation strengths appropriate for p>>n at n~100-350. '
    "Include 8-14 DISTINCT configs. A robust repeated-CV OOF search picks the winner — propose a good, diverse "
    "SPACE, not one guess. Output ONLY the JSON list."
)
DEFAULT_CLF_SPACE = ([{"model": "svc_linear", "k": k, "C": c} for k in (32, 64, 128) for c in (0.05, 0.1, 0.5)]
                     + [{"model": "logreg", "k": 64, "C": 0.1}, {"model": "rf", "k": 128},
                        {"model": "et", "k": 128}])
REG_SPACE_SYSTEM = (
    "You DESIGN the candidate REGRESSOR space for a SMALL-SAMPLE (p>>n) multimodal depression-severity "
    "task: predict a continuous clinical score, not a class. Reason from the data regime and the "
    "grounding, then output ONLY a JSON list of candidate scikit-learn configs. Each config: "
    '{"model": "ridge"|"svr_linear"|"svr_rbf"|"gbr"|"rfr"|"etr", '
    '"k": 32|64|128|256 (SelectKBest by f_regression), "alpha": 1..300 (ridge), "C": 0.03..3.0 (svr), '
    '"eps": 0.5..3.0 (svr), "lr": 0.03..0.1 (gbr), "depth": 1..3 (gbr)}. '
    "THE REGIME IS THE WHOLE PROBLEM HERE. At n~150-350 with p in the hundreds or thousands, an "
    "under-regularised fit interpolates the training scores and returns a NEGATIVE out-of-sample R2 — "
    "worse than predicting the training mean for every subject. The published baselines on this task "
    "sit exactly there (R2 from -0.08 to -0.46, CCC within +/-0.05 of zero), so a space that leans "
    "toward strong shrinkage is not timidity, it is the only region where anything beats a constant. "
    "Span shrinkage strengths across at least two orders of magnitude, and include at least one very "
    "heavily regularised candidate. Include 8-14 DISTINCT configs. Output ONLY the JSON list."
)
DEFAULT_REG_SPACE = ([{"model": "ridge", "k": k, "alpha": a} for k in (32, 128) for a in (10.0, 100.0)]
                     + [{"model": "svr_linear", "k": 64, "C": 0.3, "eps": 1.0},
                        {"model": "svr_rbf", "k": 64, "C": 1.0, "eps": 1.0},
                        {"model": "gbr", "k": 64, "lr": 0.05, "depth": 2},
                        {"model": "rfr", "k": 128}, {"model": "etr", "k": 128}])
CLF_MIN_COVERAGE = 0.5


CLF_MODELS = ("svc_linear", "svc_rbf", "logreg", "gb", "rf", "et")
REG_MODELS = ("ridge", "svr_linear", "svr_rbf", "gbr", "rfr", "etr")


def _parse_cfg_list(txt, allowed=CLF_MODELS):
    import json as _json
    import re as _re
    txt = txt or ""
    spans = _re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", txt, _re.S)
    spans += [txt[i:] for i in sorted((m.start() for m in _re.finditer(r"\[", txt)), reverse=True)]
    dec = _json.JSONDecoder()
    for sp in spans:
        try:
            obj, _ = dec.raw_decode(sp.strip())
        except Exception:
            continue
        if not isinstance(obj, list):
            continue
        out, seen = [], set()
        for c in obj:
            if not isinstance(c, dict) or c.get("model") not in allowed:
                continue
            key = _json.dumps(c, sort_keys=True)
            if key not in seen:
                seen.add(key); out.append(c)
        if out:
            return out
    return []


REG_SPACE_SYSTEM = (
    "You DESIGN the candidate REGRESSOR space for a SMALL-SAMPLE (p>>n) multimodal depression-severity "
    "task: predict a continuous clinical score, not a class. Reason from the data regime and the "
    "grounding, then output ONLY a JSON list of candidate scikit-learn configs. Each config: "
    '{"model": "ridge"|"svr_linear"|"svr_rbf"|"gbr"|"rfr"|"etr", '
    '"k": 32|64|128|256 (SelectKBest by f_regression), "alpha": 1..300 (ridge), "C": 0.03..3.0 (svr), '
    '"eps": 0.5..3.0 (svr), "lr": 0.03..0.1 (gbr), "depth": 1..3 (gbr)}. '
    "THE REGIME IS THE WHOLE PROBLEM HERE. At n~150-350 with p in the hundreds or thousands, an "
    "under-regularised fit interpolates the training scores and returns a NEGATIVE out-of-sample R2 — "
    "worse than predicting the training mean for every subject. The published baselines on this task "
    "sit exactly there (R2 from -0.08 to -0.46, CCC within +/-0.05 of zero), so a space that leans "
    "toward strong shrinkage is not timidity, it is the only region where anything beats a constant. "
    "Span shrinkage strengths across at least two orders of magnitude, and include at least one very "
    "heavily regularised candidate. Include 8-14 DISTINCT configs. Output ONLY the JSON list."
)
DEFAULT_REG_SPACE = ([{"model": "ridge", "k": k, "alpha": a} for k in (32, 128) for a in (10.0, 100.0)]
                     + [{"model": "svr_linear", "k": 64, "C": 0.3, "eps": 1.0},
                        {"model": "svr_rbf", "k": 64, "C": 1.0, "eps": 1.0},
                        {"model": "gbr", "k": 64, "lr": 0.05, "depth": 2},
                        {"model": "rfr", "k": 128}, {"model": "etr", "k": 128}])
CLF_MIN_COVERAGE = 0.5


CLF_MODELS = ("svc_linear", "svc_rbf", "logreg", "gb", "rf", "et")
REG_MODELS = ("ridge", "svr_linear", "svr_rbf", "gbr", "rfr", "etr")


def _sk(name):
    from sklearn import metrics as _m
    return getattr(_m, name)


def _ccc(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    vt, vp = yt.var(), yp.var()
    den = vt + vp + (yt.mean() - yp.mean()) ** 2
    if den <= 0:
        return 0.0
    return float(2 * np.cov(yt, yp, bias=True)[0, 1] / den)


METRICS = {
    "accuracy": lambda yt, yp: _sk("accuracy_score")(yt, yp),
    "balanced_accuracy": lambda yt, yp: _sk("balanced_accuracy_score")(yt, yp),
    "macro_f1": lambda yt, yp: _sk("f1_score")(yt, yp, average="macro", zero_division=0),
    "weighted_f1": lambda yt, yp: _sk("f1_score")(yt, yp, average="weighted", zero_division=0),
    "cohen_kappa": lambda yt, yp: _sk("cohen_kappa_score")(yt, yp),
    "official_composite": lambda yt, yp: (
        (_sk("accuracy_score")(yt, yp) + _sk("balanced_accuracy_score")(yt, yp)) / 2
        + (_sk("f1_score")(yt, yp, average="macro", zero_division=0)
           + _sk("f1_score")(yt, yp, average="weighted", zero_division=0)) / 2) / 2,
    "ccc": _ccc,
    "r2": lambda yt, yp: _sk("r2_score")(yt, yp),
    "neg_rmse": lambda yt, yp: -float(np.sqrt(_sk("mean_squared_error")(yt, yp))),
    "neg_mae": lambda yt, yp: -_sk("mean_absolute_error")(yt, yp),
}

REGRESSION_METRICS = {"ccc", "r2", "neg_rmse", "neg_mae"}


def metric_value(name, yt, yp):
    fn = METRICS.get(name, METRICS["macro_f1"])
    cast = float if name in REGRESSION_METRICS else int
    return float(fn(np.asarray(yt, cast), np.asarray(yp, cast)))


NL = chr(10)
NL2 = NL + NL

CLF_METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "cohen_kappa",
                    "official_composite")
REG_METRIC_NAMES = ("ccc", "r2", "neg_rmse", "neg_mae")

EVAL_METRIC_SYSTEM = (
    "You are handed a clinical multimodal dataset. Determine WHAT THE PUBLISHED LITERATURE ON THIS "
    "CORPUS REPORTS as its headline evaluation metric. This is a question about the benchmark's "
    "convention — not about your preference, and not about what would make a model look good.\n"
    "It matters because the metric a search OPTIMISES must be the metric the work is JUDGED by. "
    "Optimising an accuracy-flavoured score on a corpus with heavy class imbalance selects models "
    "that predict the base rate, which then lose on the balanced metric the field reports.\n"
    "YOU MUST SEARCH before answering. The convention is a published fact about a specific benchmark, "
    "and recall of it is not reliable: asked about one corpus four times this step returned two "
    "different metrics, and the majority-dominated one then collapsed a 5-class task onto the "
    "majority label while still looking fine on the metric it had been chosen by. Find the paper or "
    "challenge page that defines the corpus and read what its results table reports. Begin with "
    "`SEARCHED: yes` only if the search actually returned something you could read. If it returned "
    "nothing usable - no results, a quota or tool limit, a fetch you were not allowed to make - then "
    "begin with `SEARCHED: no` and answer from recall. Both are acceptable answers; what is not "
    "acceptable is reporting a search you could not complete as though it had confirmed something, "
    "because an unverified answer is kept for the current run only and asked again next time, and "
    "that distinction is what the first word tells us. "
    "Answer on the FIRST line exactly: `METRIC: <name> - <why the field reports this one>` where "
    "<name> is one of the OPTIONS listed in the prompt. You may not define a metric of your own; "
    "note that `official_composite` means the challenge score ((acc+bal_acc)/2 + (macro_f1+"
    "weighted_f1)/2)/2 that some of these corpora publish as their headline column - name it when "
    "that IS what the corpus reports, and do not name it merely because it looks comprehensive. "
    "name one of the options given. Note on the regression options: they are written so that HIGHER "
    "IS BETTER, so an error metric appears negated (neg_rmse). Naming it still means the field "
    "reports RMSE - the sign is only the search convention."
)


def propose_eval_metric(proposer, ctx, task_desc, report, freeze_dir, traj=None,
                        kind="classification", corpus=None):
    import hashlib
    import json as _json
    safe = ctx.replace("/", "_").replace("\\", "_")
    reg = kind == "regression"
    options = REG_METRIC_NAMES if reg else CLF_METRIC_NAMES
    fallback = "ccc" if reg else "macro_f1"
    safe = (corpus or ctx.split("/")[0]).replace("/", "_").replace("\\", "_")
    fp = os.path.join(freeze_dir, "eval_metric_%s_%s.json" % (safe, "reg" if reg else "clf"))
    if os.path.exists(fp):
        obj = _json.load(open(fp))
        print("  [eval-metric] reused FROZEN metric: %s (%s)" % (obj["metric"], fp), flush=True)
        if traj is not None:
            traj.log("eval_metric_frozen", dataset=ctx, choice=obj["metric"])
        return obj["metric"]
    from merid.llm.retrieval import decide_with_optional_search
    prompt = ("Corpus: %s" + NL2 + "TASK: %s" + NL2 + "OBSERVED INSPECTION:" + NL + "%s" + NL2
              + "OPTIONS: %s" + NL2 + "Name the metric this corpus is judged by."
              ) % (ctx, task_desc, report[:1800], " | ".join(options))
    txt = decide_with_optional_search(proposer, EVAL_METRIC_SYSTEM, prompt, traj=traj, ctx=ctx,
                                      tag="eval-metric")
    m = re.search(r"METRIC:\s*([a-z_0-9]+)", txt or "", re.I)
    name = m.group(1).lower() if m else ""
    if name not in options:
        print("  [eval-metric] no usable answer (LLM error or unparsable) -> using %s for THIS "
              "run WITHOUT freezing; the next run will ask again" % fallback, flush=True)
        return fallback
    verified = re.search(r"SEARCHED:\s*yes", txt or "", re.I) is not None
    if not verified:
        print("  [eval-metric] %s (UNVERIFIED - the search returned nothing usable, so this is "
              "recall). Using it for THIS run without freezing; the next run asks again." % name,
              flush=True)
        if traj is not None:
            traj.log("eval_metric_unverified", dataset=ctx, choice=name,
                     reasoning=(txt or "")[:3000])
        return name
    os.makedirs(freeze_dir, exist_ok=True)
    payload = {"ctx": ctx, "metric": name, "verified": True, "reasoning": (txt or "")[:2000]}
    payload["sha256"] = hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with open(fp, "w") as fh:
        _json.dump(payload, fh, indent=2)
    first = (txt or "").strip().split("\n")[0][:150]
    print("  [eval-metric] %s -> FROZEN  (%s)" % (name, first), flush=True)
    if traj is not None:
        traj.log("eval_metric", dataset=ctx, choice=name, reasoning=(txt or "")[:2000],
                 sha256=payload["sha256"])
    return name


REGIME_MARKER = "MEASURED REGIME:" + chr(10)


def _regime_key(prof):
    text = prof if isinstance(prof, str) else str(prof)
    return text.rsplit(REGIME_MARKER, 1)[-1].strip()


def _regime_facts(prof):
    import re as _re
    text = _regime_key(prof)
    facts = {}
    for name in ("n_train", "n_dev", "n_test", "p"):
        m = _re.search(r"\b%s\s*=\s*(\d+)" % name, text)
        if m:
            facts[name] = int(m.group(1))
    braces = _re.search(r"\{([^}]*)\}", text)
    if braces:
        pairs = _re.findall(r'"?(-?\d+)"?\s*:\s*(\d+)', braces.group(1))
        if pairs:
            facts["class_dist"] = {int(k): int(v) for k, v in pairs}
    m = _re.search(r"target mean\s*=\s*(-?[\d.]+)\s+sd\s*=\s*(-?[\d.]+)", text)
    if m:
        facts["target_moments"] = (round(float(m.group(1)), 4), round(float(m.group(2)), 4))
    return facts


def _freeze_matches_regime(obj, prof, fp, tag):
    was = obj.get("regime")
    if was is None:
        print(f"  [{tag}] frozen proposal records no regime — cannot verify it was made for this "
              f"data ({fp})", flush=True)
        return True
    if _regime_key(was) == _regime_key(prof):
        return True
    mine, theirs = _regime_facts(prof), _regime_facts(was)
    shared = sorted(set(mine) & set(theirs))
    if not {"n_train", "p"} <= set(shared):
        print(f"  [{tag}] frozen proposal records a regime this run cannot compare against "
              f"(read {sorted(theirs)} vs {sorted(mine)}) — keeping it ({fp})", flush=True)
        return True
    differing = [k for k in shared if mine[k] != theirs[k]]
    if not differing:
        print(f"  [{tag}] frozen proposal matches this data regime ({', '.join('%s=%s' % (k, mine[k]) for k in shared)})", flush=True)
        return True
    print(f"  [{tag}] frozen proposal was made for a DIFFERENT regime — re-proposing", flush=True)
    for key in differing:
        print(f"      {key}: frozen {theirs[key]} vs now {mine[key]}", flush=True)
    os.replace(fp, fp + ".superseded")
    return False


def propose_clf_space(proposer, prof, ctx, freeze_dir, traj=None, kind="classification"):
    import hashlib
    import json as _json
    safe = ctx.replace("/", "_").replace("\\", "_")
    reg = kind == "regression"
    system = REG_SPACE_SYSTEM if reg else CLF_SPACE_SYSTEM
    allowed = REG_MODELS if reg else CLF_MODELS
    default = DEFAULT_REG_SPACE if reg else DEFAULT_CLF_SPACE
    tag = "reg-space" if reg else "clf-space"
    fp = os.path.join(freeze_dir, f"{'reg' if reg else 'clf'}_space_{safe}.json")
    if os.path.exists(fp) and _freeze_matches_regime(_json.load(open(fp)), prof, fp, tag):
        obj = _json.load(open(fp))
        print(f"  [{tag}] reused {len(obj['configs'])} FROZEN configs ({fp})", flush=True)
        if traj is not None:
            traj.log(f"{tag.replace(chr(45), chr(95))}_frozen", dataset=ctx, choice=obj["configs"], sha256=obj.get("sha256"))
        return obj["configs"]
    from merid.llm.retrieval import decide_with_optional_search
    txt = decide_with_optional_search(proposer, system,
                                      f"OBSERVED DATA REGIME (measured from THIS data): {prof}\n\n"
                                      f"Design the candidate classifier list now.",
                                      traj=traj, ctx=ctx, tag=tag)
    cfgs = _parse_cfg_list(txt, allowed)
    if not cfgs:
        print(f"  [{tag}] proposal empty/unparsable -> default space ({len(default)}) "
              f"for THIS run, not frozen", flush=True)
        if traj is not None:
            traj.log(f"{tag.replace(chr(45), chr(95))}_unparsable", dataset=ctx,
                     reasoning=(txt or "")[:3000], used="default_space")
        return default
    os.makedirs(freeze_dir, exist_ok=True)
    payload = {"ctx": ctx, "regime": prof, "configs": cfgs, "reasoning": txt}
    payload["sha256"] = hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with open(fp, "w") as fh:
        _json.dump(payload, fh, indent=2)
    print(f"  [{tag}] agent proposed {len(cfgs)} configs -> FROZEN {fp}", flush=True)
    if traj is not None:
        traj.log(f"{tag.replace(chr(45), chr(95))}_propose", dataset=ctx, reasoning=txt, choice=cfgs, sha256=payload["sha256"])
    return cfgs


MODALITY_PRIOR_SYSTEM = (
    "You judge, BEFORE any model is fitted, how much each enabled feature source is likely to "
    "contribute to THIS prediction task, and how confident you are in that judgement.\n"
    "You are given the OBSERVED INSPECTION (which extractors exist, their shapes, and their VALUE "
    "STATISTICS — mean/std/range/zero-fraction) and the MEASURED REGIME (n, subjects, p, class "
    "balance). You are NOT given any accuracy or OOF result: this is a prior, stated before the "
    "evidence, and it will be checked against a leave-one-out ablation afterwards.\n"
    "Reason from what is actually observable: a source whose values span an absurd numeric range or "
    "are mostly zeros is suspect regardless of how standard it is; a very high-dimensional source at "
    "n~300 risks contributing variance rather than signal; a low-dimensional structured source may "
    "carry little but be cheap and stable. Do NOT rank sources by their reputation in the literature "
    "— on this dataset a widely-used extractor was numerically unusable and scored at chance.\n"
    "Output ONLY a JSON object mapping each source name to "
    '{"weight": 0.0..1.0, "confidence": "high"|"low", "reason": "<one clause tied to an observed number>"}. '
    "`weight` = expected share of the task's predictive signal. `confidence` = 'high' only if the "
    "observation itself settles it; say 'low' when you are extrapolating — a low-confidence source "
    "gets the expensive permutation test, so an honest 'low' costs you nothing and a false 'high' "
    "skips a check that would have caught you."
)


def _parse_modality_priors(txt, sources):
    import json as _json
    obj = None
    txt = txt or ""
    cands = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.S)
    for start in sorted((m.start() for m in re.finditer(r"\{", txt)), reverse=True):
        cands.append(txt[start:])
    for c in cands:
        dec = _json.JSONDecoder()
        try:
            o, _ = dec.raw_decode(c.strip())
        except Exception:
            continue
        if isinstance(o, dict) and any(k in sources for k in o):
            obj = o
            break
    if obj is None:
        return {}
    out = {}
    for k, v in obj.items() if isinstance(obj, dict) else []:
        if k not in sources or not isinstance(v, dict):
            continue
        try:
            w = float(v.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        out[k] = {"weight": max(0.0, min(1.0, w)),
                  "confidence": "high" if str(v.get("confidence", "")).lower() == "high" else "low",
                  "reason": str(v.get("reason", ""))[:200]}
    return out


def propose_modality_priors(proposer, report, prof, recipe, ctx, freeze_dir, traj=None):
    import hashlib
    import json as _json
    sources = [k for k, _ in _modalities_of(recipe)]
    if not sources:
        return {}
    safe = ctx.replace("/", "_").replace("\\", "_")
    fp = os.path.join(freeze_dir, f"modality_prior_{safe}.json")
    if os.path.exists(fp):
        obj = _json.load(open(fp))
        was = sorted(k for k in ("audio", "video", "text", "ind_emb", "meta")
                     if (obj.get("recipe") or {}).get(k))
        if was == sorted(sources):
            print(f"  [modality-prior] reused FROZEN priors ({fp})", flush=True)
            return obj.get("priors", {})
        print(f"  [modality-prior] frozen prior was stated for sources {was} but this recipe enables "
              f"{sorted(sources)} — re-stating", flush=True)
        os.replace(fp, fp + ".superseded")
    from merid.llm.retrieval import decide_with_optional_search
    prompt = (f"Dataset: {ctx}\n\nOBSERVED INSPECTION (extractors, shapes, value statistics):\n"
              f"{report[:2400]}\n\nMEASURED REGIME: {prof}\n\n"
              f"ENABLED FEATURE SOURCES in the chosen recipe: {sources}\n"
              f"(recipe = {recipe})\n\n"
              f"State your prior for each source now, as the JSON object described.")
    txt = decide_with_optional_search(proposer, MODALITY_PRIOR_SYSTEM, prompt, traj=traj, ctx=ctx,
                                      tag="modality-prior")
    priors = _parse_modality_priors(txt, sources)
    if not priors:
        print(f"  [modality-prior] unparsable -> every source will get the full test", flush=True)
    os.makedirs(freeze_dir, exist_ok=True)
    payload = {"ctx": ctx, "recipe": recipe, "priors": priors, "reasoning": (txt or "")[:3000]}
    payload["sha256"] = hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with open(fp, "w") as fh:
        _json.dump(payload, fh, indent=2)
    for k, v in priors.items():
        print(f"  [modality-prior] {k}: weight={v['weight']:.2f} conf={v['confidence']} "
              f"— {v['reason'][:70]}", flush=True)
    if traj is not None:
        traj.log("modality_prior", dataset=ctx, choice=priors, reasoning=(txt or "")[:3000],
                 sha256=payload["sha256"])
    return priors


CODEGEN_SYSTEM = (
    "You decide whether the STANDARD scikit-learn families (linear/RBF SVM, logistic regression, "
    "gradient boosting, random forest, extra trees — each behind VarianceThreshold + StandardScaler + "
    "SelectKBest) are enough for the observed regime, or whether a method they cannot express would suit "
    "it better.\n"
    "Answer on the FIRST line: `CODEGEN: no — <why the standard families already cover this regime>` or "
    "`CODEGEN: yes — <the specific property of THIS regime they cannot express>`.\n"
    "Only if yes: output ONE ```python``` block defining `build()` that returns a fresh, UNFITTED "
    "scikit-learn-compatible estimator (needs .fit(X, y) and .predict(X)). Compose it from sklearn/numpy "
    "only — Pipeline, feature selection, decomposition (PLS/PCA), calibration, custom BaseEstimator, "
    "ensembling. It must be self-contained, take no arguments, and be cheap enough for repeated CV. "
    "Do NOT reach for a deep network: n is a few hundred.\n"
    "Whatever you produce will be scored by the SAME leak-free repeated per-subject OOF as every standard "
    "candidate and only used if it actually wins — so propose it because the regime calls for it, not to "
    "look sophisticated. Saying `no` is a perfectly good answer."
)


def propose_codegen_method(proposer, prof, ctx, freeze_dir, traj=None):
    import hashlib
    import json as _json
    from merid.core.coupled_pipeline_exploration.operation import (
        build_from_code, extract_code, smoke_test)
    from merid.llm.retrieval import decide_with_optional_search
    safe = ctx.replace("/", "_").replace("\\", "_")
    fp = os.path.join(freeze_dir, f"codegen_{safe}.json")
    if os.path.exists(fp) and _freeze_matches_regime(_json.load(open(fp)), prof, fp, "codegen"):
        obj = _json.load(open(fp))
        if not obj.get("code"):
            print(f"  [codegen] reused FROZEN decision: agent declined for this dataset", flush=True)
            if traj is not None:
                traj.log("codegen_frozen", dataset=ctx, choice="declined")
            return None, None
        try:
            build, code = build_from_code(obj["code"])
            print(f"  [codegen] reused FROZEN build() ({fp})", flush=True)
            if traj is not None:
                traj.log("codegen_frozen", dataset=ctx, choice="build")
            return build, code
        except Exception as exc:
            print(f"  [codegen] frozen build failed ({type(exc).__name__}) -> ignoring", flush=True)
            return None, None
    txt = decide_with_optional_search(proposer, CODEGEN_SYSTEM,
                                      f"OBSERVED DATA REGIME (measured from THIS data): {prof}\n\n"
                                      f"Decide, and if yes write build().",
                                      traj=traj, ctx=ctx, tag="codegen")
    code, build = None, None
    if txt and re.search(r"CODEGEN:\s*yes", txt, re.I):
        snippet = extract_code(txt)
        if snippet and "def build" in snippet:
            try:
                build, code = build_from_code(snippet)
                if not smoke_test(build, True):
                    print("  [codegen] build() failed smoke test -> discarded", flush=True)
                    build, code = None, None
            except Exception as exc:
                print(f"  [codegen] build() error: {type(exc).__name__}: {str(exc)[:120]} -> discarded",
                      flush=True)
                build, code = None, None
    os.makedirs(freeze_dir, exist_ok=True)
    payload = {"ctx": ctx, "regime": prof, "code": code, "decision": (txt or "")[:2000]}
    payload["sha256"] = hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with open(fp, "w") as fh:
        _json.dump(payload, fh, indent=2)
    print(f"  [codegen] {'build() accepted -> enters the OOF pool' if code else 'no method codegen d'}"
          f" -> FROZEN {fp}", flush=True)
    if traj is not None:
        traj.log("codegen_propose", dataset=ctx, reasoning=(txt or "")[:3000], code=code,
                 produced=code is not None, sha256=payload["sha256"])
    return build, code


CRED_N_PERM = 20
CRED_Z_SIGNAL = 2.0
MOD_MARGIN_FLOOR = 0.01


def _permute_subject_labels(y, groups, rng):
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    lab = np.array([y[groups == g][0] for g in uniq])
    perm = rng.permutation(lab)
    m = dict(zip(uniq, perm))
    return np.asarray([m[g] for g in groups], dtype=int)


def _official_oof(make, X, y, groups, seed=0):
    oof, _, _ = _oof(make, X, X[:1], y, grouped_folds(y, groups, seed=seed))
    mk = np.asarray(oof) >= 0
    if mk.mean() < CLF_MIN_COVERAGE:
        return None
    return metric_value(SEL_METRIC, y[mk], oof[mk])


def _modalities_of(recipe):
    mods = []
    for k in ("audio", "video", "text"):
        if recipe.get(k):
            mods.append((k, recipe[k]))
    for k in ("ind_emb", "meta"):
        if recipe.get(k):
            mods.append((k, True))
    return mods


def modality_credibility(load, recipe, y, groups, make=None, n_perm=CRED_N_PERM, seed=0,
                         priors=None):
    make = make or (lambda p: make_clf(PROBE_CFG, p))
    y = np.asarray(y, int)
    groups = np.asarray(groups)

    def oof_reps(rc):
        try:
            X = np.asarray(load(rc)["train_X"], float)
        except Exception:
            return None
        if X.ndim != 2 or X.shape[1] == 0:
            return None
        vals = [_official_oof(make, X, y, groups, seed=s) for s in range(RECIPE_CV_REPEATS)]
        vals = [v for v in vals if v is not None]
        return vals or None

    full_reps = oof_reps(recipe)
    full = float(np.mean(full_reps)) if full_reps else None
    if full is None:
        return None
    out = {"_full_oof": round(float(full), 4), "modalities": {}}

    priors = priors or {}
    for k, _v in _modalities_of(recipe):
        ablated = dict(recipe)
        ablated[k] = None if k in ("audio", "video") else False
        loo_reps = oof_reps(ablated)
        pri = priors.get(k, {})
        if loo_reps is None:
            out["modalities"][k] = {"loo": None, "marginal": None, "verdict": "UNTESTABLE",
                                    "prior_weight": pri.get("weight"), "prior_conf": pri.get("confidence")}
            continue
        n = min(len(full_reps), len(loo_reps))
        diffs = [full_reps[i] - loo_reps[i] for i in range(n)]
        marg = float(np.mean(diffs))
        marg_sd = float(np.std(diffs))
        loo = float(np.mean(loo_reps[:n]))
        significant = abs(marg) > max(MOD_MARGIN_FLOOR, marg_sd)
        if not significant:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "SIGNAL" if marg > 0 else "NOISE"
        out["modalities"][k] = {
            "full": round(float(full), 4), "loo": round(loo, 4),
            "marginal": round(marg, 4), "marginal_sd": round(marg_sd, 4), "n_repeats": n,
            "verdict": verdict,
            "prior_weight": pri.get("weight"), "prior_conf": pri.get("confidence"),
        }

    disagrees = any(
        (priors.get(k, {}).get("confidence") == "high")
        and v.get("verdict") in ("SIGNAL", "NOISE")
        and ((priors[k]["weight"] >= 0.25) != (v["verdict"] == "SIGNAL"))
        for k, v in out["modalities"].items()
    )
    low_conf = any(p.get("confidence") == "low" for p in priors.values())
    untestable = any(v.get("marginal") is None for v in out["modalities"].values())
    need_null = (not priors) or low_conf or disagrees or untestable
    out["null_triage"] = {"ran": bool(need_null), "low_confidence_source": bool(low_conf),
                          "prior_contradicted": bool(disagrees), "untestable_source": bool(untestable),
                          "reason": ("a source could not be ablated (single-source recipe) -> null is "
                                     "the only available evidence" if untestable else
                                     "no usable prior" if not priors else
                                     "agent flagged low confidence" if low_conf else
                                     "measured marginal contradicts a high-confidence prior" if disagrees
                                     else "agent confident and marginals agree -> null skipped")}
    if not need_null:
        out["null"] = {"verdict": "SKIPPED",
                       "note": "permutation null not spent: agent was confident on every source and "
                               "every leave-one-out marginal agreed with its prior"}
        out["calibration"] = _prior_calibration(out["modalities"], priors)
        return out

    rng = np.random.RandomState(seed)
    X_full = np.asarray(load(recipe)["train_X"], float)
    null = [s for s in (_official_oof(make, X_full, _permute_subject_labels(y, groups, rng),
                                      groups, seed=seed)
                        for _ in range(n_perm)) if s is not None]
    if len(null) >= 5:
        nm, ns = float(np.mean(null)), float(np.std(null))
        z = (full - nm) / max(ns, 1e-6)
        task_signal = z >= CRED_Z_SIGNAL
        out["null"] = {"real": round(float(full), 4), "null_mean": round(nm, 4),
                       "null_std": round(ns, 4), "z": round(z, 2), "n_perm": len(null),
                       "verdict": "SIGNAL" if task_signal else "NOISE"}
        if not task_signal:
            for k in out["modalities"]:
                out["modalities"][k]["verdict"] = "UNRELIABLE"
            out["note"] = ("task OOF is inside its permutation null -> no signal to attribute; "
                           "per-modality marginals are noise and are NOT interpretable")
    else:
        out["null"] = {"verdict": "UNTESTABLE", "n_perm": len(null)}
        for k in out["modalities"]:
            out["modalities"][k]["verdict"] = "UNRELIABLE"
    out["calibration"] = _prior_calibration(out["modalities"], priors)
    return out


def _prior_calibration(measured, priors):
    rows, hit, tested = {}, 0, 0
    for k, v in measured.items():
        p = priors.get(k)
        if not p or v.get("verdict") not in ("SIGNAL", "NOISE"):
            rows[k] = {"prior": (p or {}).get("weight"), "measured": v.get("marginal"),
                       "verdict": v.get("verdict"), "agree": None}
            continue
        predicted_useful = p["weight"] >= 0.25
        measured_useful = v["verdict"] == "SIGNAL"
        agree = predicted_useful == measured_useful
        tested += 1
        hit += int(agree)
        rows[k] = {"prior": p["weight"], "prior_conf": p["confidence"],
                   "measured": v["marginal"], "predicted_useful": predicted_useful,
                   "measured_useful": measured_useful, "agree": agree}
    return {"per_source": rows, "n_tested": tested, "n_agree": hit,
            "accuracy": (round(hit / tested, 3) if tested else None)}


def signal_credibility(make, X, y, groups, n_perm=CRED_N_PERM, seed=0):
    real = _official_oof(make, X, y, groups, seed=seed)
    if real is None:
        return None
    rng = np.random.RandomState(seed)
    null = []
    for _ in range(n_perm):
        s = _official_oof(make, X, _permute_subject_labels(y, groups, rng), groups, seed=seed)
        if s is not None:
            null.append(s)
    if len(null) < 5:
        return None
    nm, ns = float(np.mean(null)), float(np.std(null))
    z = (real - nm) / max(ns, 1e-6)
    p = (sum(1 for s in null if s >= real) + 1) / (len(null) + 1)
    return {"real": round(float(real), 4), "null_mean": round(nm, 4), "null_std": round(ns, 4),
            "z": round(float(z), 2), "p": round(float(p), 4), "n_perm": len(null),
            "verdict": "SIGNAL" if (z >= CRED_Z_SIGNAL and p <= 0.05) else "NOISE"}


SUBJECT_CONSENSUS = os.environ.get("SUBJECT_CONSENSUS", "1") != "0"

METRIC_DECODE = os.environ.get("METRIC_DECODE", "1") != "0"
PROBA_MODELS = ("logreg", "rf", "et", "gb")
BIAS_STEPS = (-0.4, -0.25, -0.15, 0.15, 0.25, 0.4)


def _tune_bias(logp, y, classes, n_iter=300, metric=None):
    r = np.random.default_rng(0)
    b = np.zeros(len(classes))

    def off(bb):
        return metric_value(metric or SEL_METRIC, y, classes[(logp + bb).argmax(1)])

    best = off(b)
    for _ in range(n_iter):
        j = int(r.integers(len(classes)))
        cand = b.copy()
        cand[j] += float(r.choice(BIAS_STEPS))
        v = off(cand)
        if v > best:
            best, b = v, cand
    return b


def _oof_proba(cfg, X, y, groups, seed):
    classes = np.unique(np.asarray(y, int))
    P = np.zeros((len(y), len(classes)))
    cov = np.zeros(len(y), bool)
    for tr, va in grouped_folds(y, groups, seed=seed):
        if len(np.unique(y[tr])) < 2:
            continue
        m = make_clf(cfg, X.shape[1])
        m.fit(X[tr], y[tr])
        pr = m.predict_proba(X[va])
        col = {c: i for i, c in enumerate(m.classes_)}
        for j, c in enumerate(classes):
            if c in col:
                P[va, j] = pr[:, col[c]]
        cov[va] = True
    return P, cov


def _metric_decode(X, Xt, y, groups, configs):
    y = np.asarray(y, int)
    classes = np.unique(y)
    if len(classes) < 3:
        return None, None

    usable = [c for c in configs if c.get("model") in PROBA_MODELS]
    best = None
    for c in usable:
        Ps, offs, ok = [], [], True
        for s in range(RECIPE_CV_REPEATS):
            try:
                P, cov = _oof_proba(c, X, y, groups, seed=s)
            except Exception:
                ok = False
                break
            if cov.mean() < CLF_MIN_COVERAGE:
                ok = False
                break
            Ps.append((P, cov))
            offs.append(metric_value(SEL_METRIC, y[cov], classes[P[cov].argmax(1)]))
        if not ok or not offs:
            continue
        sc = float(np.mean(offs) - RECIPE_STD_PENALTY * np.std(offs))
        if best is None or sc > best[0]:
            best = (sc, c, Ps)
    if best is None:
        return None, None
    plain_score, cfg, Ps = best

    logs = [np.log(np.clip(P[cov], 1e-6, None)) for (P, cov) in Ps]
    ys = [y[cov] for (P, cov) in Ps]
    b = _tune_bias(np.vstack(logs), np.concatenate(ys), classes)
    tuned_offs = [metric_value(SEL_METRIC, y[cov],
                               classes[(np.log(np.clip(P[cov], 1e-6, None)) + b).argmax(1)])
                  for (P, cov) in Ps]
    tuned_score = float(np.mean(tuned_offs) - RECIPE_STD_PENALTY * np.std(tuned_offs))
    if tuned_score <= plain_score:
        return None, None

    m = make_clf(cfg, X.shape[1])
    m.fit(X, y)
    pr = m.predict_proba(Xt)
    col = {c: i for i, c in enumerate(m.classes_)}
    Pe = np.zeros((Xt.shape[0], len(classes)))
    for j, c in enumerate(classes):
        if c in col:
            Pe[:, j] = pr[:, col[c]]
    return tuned_score, np.asarray(classes[(np.log(np.clip(Pe, 1e-6, None)) + b).argmax(1)], int)


def _pool_by_group(pred, groups):
    pred = np.asarray(pred, int)
    groups = np.asarray(groups)
    out = pred.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = int(stats.mode(pred[m], keepdims=False).mode)
    return out


def _subject_level(y, groups):
    y = np.asarray(y)
    groups = np.asarray(groups)
    if groups is None or len(groups) != len(y):
        return False
    return all(len(np.unique(y[groups == g])) == 1 for g in np.unique(groups))


def _subject_of(seg_id):
    return str(seg_id).split("_")[0]


ORDINAL_BASES = ({"model": "logreg", "k": 64, "C": 0.1},
                 {"model": "logreg", "k": 128, "C": 0.1},
                 {"model": "logreg", "k": 64, "C": 0.5})
ORDINAL_MEMBER = os.environ.get("ORDINAL_MEMBER", "1") != "0"


class FrankHallOrdinal(BaseEstimator, ClassifierMixin):
    def __init__(self, base_cfg=None):
        self.base_cfg = base_cfg or {"model": "logreg", "k": 64, "C": 0.1}

    def fit(self, X, y):
        y = np.asarray(y, int)
        self.classes_ = np.unique(y)
        self.K_ = len(self.classes_)
        cmap = {c: i for i, c in enumerate(self.classes_)}
        yi = np.array([cmap[v] for v in y])
        p = X.shape[1]
        self.models_ = []
        self.fit_calls_ = 0
        for k in range(self.K_ - 1):
            b = (yi > k).astype(int)
            if len(np.unique(b)) < 2:
                self.models_.append(("const", float(b.mean())))
                continue
            m = make_clf(self.base_cfg, p)
            self.fit_calls_ += 1
            m.fit(X, b)
            self.models_.append(("model", m))
        return self

    def _cum(self, X):
        cols = []
        for kind, m in self.models_:
            if kind == "const":
                cols.append(np.full(X.shape[0], m))
            else:
                idx = list(m.classes_).index(1) if 1 in list(m.classes_) else None
                cols.append(m.predict_proba(X)[:, idx] if idx is not None else np.zeros(X.shape[0]))
        return np.clip(np.asarray(cols).T, 0.0, 1.0)

    def predict_proba(self, X):
        c = self._cum(X)
        n = X.shape[0]
        P = np.zeros((n, self.K_))
        P[:, 0] = 1.0 - c[:, 0]
        for k in range(1, self.K_ - 1):
            P[:, k] = c[:, k - 1] - c[:, k]
        P[:, self.K_ - 1] = c[:, self.K_ - 2]
        P = np.clip(P, 1e-9, None)
        return P / P.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def _ordinal_candidates(y):
    if not ORDINAL_MEMBER or len(np.unique(np.asarray(y, int))) < 3:
        return []
    import json as _json
    return [(f"ordinal_frankhall:{_json.dumps(b, sort_keys=True)}",
             (lambda p, b=b: FrankHallOrdinal(b))) for b in ORDINAL_BASES]


def _oof(make, X, Xt, y, folds):
    from scipy import stats
    p = X.shape[1]
    oof = np.full(len(y), -1)
    tp, err = [], ""
    for tr, va in folds:
        if len(set(y[tr].tolist())) < 2:
            continue
        try:
            m = make(p)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict(X[va])
            tp.append(m.predict(Xt))
        except Exception as exc:
            if not err:
                err = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
    if not tp:
        return oof, np.full(Xt.shape[0], int(np.bincount(y).argmax())), err or "all folds degenerate"
    return oof, np.asarray(stats.mode(np.array(tp), axis=0, keepdims=False).mode).ravel(), err


SEL_METRIC = "official_composite"


def _subject_bootstrap_se(y, yp, groups, metric, n_boot=1000, seed=0):
    y, yp, groups = np.asarray(y), np.asarray(yp), np.asarray(groups)
    subs = np.unique(groups)
    if len(subs) < 4:
        return 0.0
    idx_of = {g: np.flatnonzero(groups == g) for g in subs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(subs), len(subs))
        rows = np.concatenate([idx_of[subs[i]] for i in pick])
        if len(np.unique(y[rows])) < 2:
            continue
        out.append(metric_value(metric, y[rows], yp[rows]))
    return float(np.std(out)) if out else 0.0


def _candidate_complexity(label):
    import json as _json
    if label.startswith("codegen:"):
        return (3, 10 ** 6, 9)
    if label.startswith("ordinal_"):
        return (2, 10 ** 6, 9)
    try:
        c = _json.loads(label)
    except Exception:
        return (1, 10 ** 6, 9)
    fam = {"svc_linear": 0, "logreg": 0, "svc_rbf": 1, "gb": 2, "rf": 3, "et": 3}
    return (0, int(c.get("k", 128)), fam.get(c.get("model"), 5))


def select_clf(X, Xt, y, groups, configs, task, traj, ctx, topk=3, codegen_build=None):
    from scipy import stats
    import json as _json
    cands = [(_json.dumps(c, sort_keys=True), (lambda p, c=c: make_clf(c, p))) for c in configs]
    if codegen_build is not None:
        cands.append(("codegen:agent_written", lambda p: codegen_build()))
    cands += _ordinal_candidates(y)
    pool = SUBJECT_CONSENSUS and _subject_level(np.asarray(y, int), groups)
    scored, errs = [], []
    for label, make in cands:
        offs, covs, ok, err = [], [], True, ""
        oof0, mk0 = None, None
        for s in range(RECIPE_CV_REPEATS):
            oof, _, e = _oof(make, X, Xt, y, grouped_folds(y, groups, seed=s))
            err = err or e
            mk = np.asarray(oof) >= 0
            covs.append(float(mk.mean()))
            if covs[-1] < CLF_MIN_COVERAGE:
                ok = False; break
            yp_oof = _pool_by_group(oof[mk], np.asarray(groups)[mk]) if pool else oof[mk]
            offs.append(metric_value(SEL_METRIC, y[mk], yp_oof))
            if s == 0:
                oof0, mk0 = yp_oof, mk
        if not ok or not offs:
            print(f"    [clf {task}] {label} -> INVALID (coverage {(np.mean(covs) if covs else 0):.2f})"
                  f"{'  cause: ' + err if err else ''}", flush=True)
            errs.append(err)
            continue
        sc = float(np.mean(offs) - RECIPE_STD_PENALTY * np.std(offs))
        _, test, _ = _oof(make, X, Xt, y, grouped_folds(y, groups, seed=0))
        scored.append((sc, test, label, make, oof0, mk0))
        star = ("  <-- ORDINAL" if label.startswith("ordinal_") else
                "  <-- CODEGEN" if label.startswith("codegen:") else "")
        print(f"    [clf {task}] {label} -> robust-official={sc:.4f} "
              f"(off={np.mean(offs):.3f}±{np.std(offs):.3f} cov={np.mean(covs):.2f}){star}", flush=True)
        if traj is not None:
            traj.log("clf_select", dataset=ctx, task=task, choice=label, robust_official=round(sc, 4),
                     oof_coverage=round(float(np.mean(covs)), 3), codegen=label.startswith("codegen:"))
    if not scored:
        raise RuntimeError(
            f"[{ctx} {task}] every classifier candidate failed — this is a BUG, not a result. "
            f"First cause: {next((e for e in errs if e), 'unknown')}. "
            f"(A silent majority-class fallback here would score Macro-F1 ~0.33/0.22 and read as a real "
            f"finding.)")
    scored.sort(key=lambda z: z[0], reverse=True)
    seen, uniq, kept = set(), [], []
    for sc, test, key, _mk, _o, _m in scored:
        if key in seen:
            continue
        seen.add(key); uniq.append(test); kept.append(key)

    se = 0.0
    if scored[0][4] is not None:
        se = _subject_bootstrap_se(y[scored[0][5]], scored[0][4],
                                   np.asarray(groups)[scored[0][5]], SEL_METRIC)
    best = scored[0][0]
    tied = [k for (sc, _t, k, _mk, _o, _m) in scored if sc >= best - se and k in seen]
    tied = sorted(dict.fromkeys(tied), key=_candidate_complexity)
    if se > 0 and len(tied) > 1:
        order = {k: i for i, k in enumerate(tied)}
        pick = sorted([k for k in kept if k in order], key=lambda k: order[k])[:topk]
        print(f"    [1-SE {task}] subject-bootstrap SE={se:.4f}; {len(tied)} of {len(kept)} candidates "
              f"within 1 SE of {best:.4f} -> ensemble filled simplest-first", flush=True)
        if traj is not None:
            traj.log("clf_select_1se", dataset=ctx, task=task, se=round(se, 4), n_tied=len(tied),
                     best=round(best, 4), chosen=pick, top_by_score=kept[:topk])
    else:
        pick = kept[:topk]
        print(f"    [1-SE {task}] subject-bootstrap SE={se:.4f}; the winner stands alone inside 1 SE "
              f"-> top-{topk} by score", flush=True)
    pos = {k: i for i, k in enumerate(kept)}
    uniq = [uniq[pos[k]] for k in pick] + [uniq[pos[k]] for k in kept if k not in pick]
    kept = pick + [k for k in kept if k not in pick]
    top = kept[:topk]

    cred = signal_credibility(scored[0][3], X, y, groups)
    if cred:
        print(f"    [credibility {task}] real={cred['real']:.4f}  null={cred['null_mean']:.4f}"
              f"±{cred['null_std']:.4f}  z={cred['z']:+.2f}  p={cred['p']:.3f}  -> {cred['verdict']}"
              + ("" if cred["verdict"] == "SIGNAL"
                 else "  (train-OOF indistinguishable from chance)"), flush=True)
        if traj is not None:
            traj.log("signal_credibility", dataset=ctx, task=task, **cred)
    if codegen_build is not None:
        won = any(k.startswith("codegen:") for k in top)
        rank = next((i + 1 for i, k in enumerate(kept) if k.startswith("codegen:")), None)
        print(f"    [clf {task}] codegen method ranked {rank}/{len(kept)} -> "
              f"{'IN the top-K ensemble' if won else 'NOT used (standard families won)'}", flush=True)
        if traj is not None:
            traj.log("codegen_outcome", dataset=ctx, task=task, rank=rank, n_candidates=len(kept),
                     used=won)
    print(f"    [clf {task}] top-{len(top)} of {len(uniq)} distinct -> ensemble", flush=True)
    yp = np.asarray(stats.mode(np.array(uniq[:topk]), axis=0, keepdims=False).mode).ravel()

    best_oof = float(scored[0][0])
    return yp, pool, best_oof


def write_bundle(a, rows, records):
    import hashlib
    import json as _json
    import platform
    import sys as _sys
    d = a.artifact_dir
    os.makedirs(os.path.join(d, "adapters"), exist_ok=True)
    metrics = [{"track": t, "window": w, "task": tk, "n": n, "accuracy": acc, "macroF1": mf1,
                "weightedAccuracy": wacc, "weightedF1": wf1, "taskAccuracy": (acc + wacc) / 2,
                "taskF1": (mf1 + wf1) / 2, "official": off}
               for t, w, tk, n, acc, wacc, mf1, wf1, off in rows]
    by_track = {}
    for m in metrics:
        by_track.setdefault(m["track"], []).append(m)
    track_scores = {k: {"n_tasks": len(v),
                        "track_accuracy": float(np.mean([x["taskAccuracy"] for x in v])),
                        "track_f1": float(np.mean([x["taskF1"] for x in v])),
                        "track_score": float(np.mean([x["official"] for x in v]))}
                    for k, v in by_track.items()}
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        _json.dump({"metrics": metrics, "track_scores": track_scores}, fh, indent=2)
    with open(os.path.join(d, "config.json"), "w") as fh:
        _json.dump({"args": vars(a), "python": _sys.version, "platform": platform.platform()}, fh, indent=2)
    for ctx, rec in records.items():
        code = rec.pop("code", None)
        if code:
            with open(os.path.join(d, "adapters", f"{ctx.replace('/', '_')}.py"), "w") as fh:
                fh.write(code)
    with open(os.path.join(d, "run_records.json"), "w") as fh:
        _json.dump(records, fh, indent=2, default=str)
    try:
        import scipy
        import sklearn
        env = (f"python {_sys.version}\nnumpy {np.__version__}\nsklearn {sklearn.__version__}\n"
               f"scipy {scipy.__version__}\n")
    except Exception:
        env = f"python {_sys.version}\nnumpy {np.__version__}\n"
    with open(os.path.join(d, "environment.txt"), "w") as fh:
        fh.write(env)
    print(f"\n[artifacts] bundle -> {d}/  (metrics.json, config.json, run_records.json, adapters/*.py, "
          f"environment.txt); recipe spaces in {a.freeze_dir}/", flush=True)


def classical_route(data, tasks, proposer, prof, ctx, traj, rows, track, window, clf_configs,
                    codegen_build=None):
    X = np.asarray(data["train_X"], float); Xt = np.asarray(data["test_X"], float)
    groups = data.get("train_subject")
    test_ids = data.get("test_ids")
    test_groups = np.array([_subject_of(i) for i in test_ids]) if test_ids is not None else None
    for task, _lk in tasks:
        trk, tek = f"train_{task}", f"test_{task}"
        if trk not in data or tek not in data:
            continue
        y = np.asarray(data[trk], int)
        if len(set(y.tolist())) < 2:
            continue
        yp, pool, std_oof = select_clf(X, Xt, y, groups, clf_configs, task, traj, ctx,
                                       codegen_build=codegen_build)
        if METRIC_DECODE:
            dec_oof, dec_pred = _metric_decode(X, Xt, y, groups, clf_configs)
            if dec_pred is not None and dec_oof is not None and dec_oof > std_oof:
                print(f"    [decode {task}] metric-aware robust-oof={dec_oof:.4f} > standard "
                      f"{std_oof:.4f} -> DEPLOYED", flush=True)
                yp = dec_pred
                if traj is not None:
                    traj.log("metric_decode", dataset=ctx, task=task, used=True,
                             decode_oof=round(dec_oof, 4), standard_oof=round(std_oof, 4))
            elif dec_oof is not None:
                print(f"    [decode {task}] metric-aware robust-oof={dec_oof:.4f} <= standard "
                      f"{std_oof:.4f} -> standard kept", flush=True)
        pooled = pool and test_groups is not None and len(test_groups) == len(yp)
        if pooled:
            yp_raw = np.asarray(yp, int)
            yp = _pool_by_group(yp_raw, test_groups)
            n_rev = int(np.sum(yp != yp_raw))
            print(f"  [subject-consensus {task}] pooled {len(yp)} test segments over "
                  f"{len(np.unique(test_groups))} subjects ({n_rev} revised to subject vote)", flush=True)
        acc, wacc, mf1, wf1, off = official(data[tek], yp)
        rows.append((track, window, task, len(yp), acc, wacc, mf1, wf1, off))
        traj.log("result", dataset=ctx, task=task,
                 observation={"official": round(off, 4), "macroF1": round(mf1, 4), "subject_pooled": bool(pooled)})
        print(f"  {task:<9} Acc={acc:.4f} MF1={mf1:.4f} Official={off:.4f}"
              f"{'  [subject-pooled]' if pooled else ''}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--test_root", default=None)
    ap.add_argument("--llm_model", default="claude-opus-4-8")
    ap.add_argument("--proposer", default="anthropic", choices=["anthropic"])
    ap.add_argument("--llm_api_key", default=None)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--freeze_dir", default="artifacts/recipe_space",
                    help="where the agent-proposed recipe space is frozen (reused on re-run for reproducibility)")
    ap.add_argument("--artifact_dir", default="artifacts/run",
                    help="auditable bundle: metrics.json, config.json, adapters/*.py + sha256, environment.txt")
    ap.add_argument('--evaluation', choices=['development', 'final'], default='final')
    ap.add_argument('--ctx', default='MPDD-Young/1s')
    ap.add_argument('--task', default='binary')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--development_folds', type=int, default=3)
    ap.add_argument('--max_evaluations', type=int, default=256)
    ap.add_argument('--review_checks', type=int, default=2)
    ap.add_argument('--promotion_se', type=float, default=0.0,
                    help='incumbent rule: a challenger must beat the incumbent by this many subject-bootstrap '
                         'standard errors of the incumbent score (0 = any margin above 1e-12)')
    ap.add_argument('--retrospective', type=int, default=1)
    ap.add_argument('--search_mode',
                    choices=['joint', 'staged', 'fixed_recipe', 'fixed_model', 'no_search', 'no_cpe'],
                    default='joint',
                    help='joint = every recipe x model pair; staged = probe-ranked recipes then '
                         'models; fixed_recipe = carry the warm-start recipe and search models '
                         'only; fixed_model = hold the probe estimator and search recipes only; '
                         'no_search = hold both at those neutral values, one candidate; '
                         'no_cpe = no exploration, the improver drives from the incumbent')
    ap.add_argument('--reliability_gate', choices=['on', 'off'], default='on',
                    help="measure whether the development ranking transfers to a held-out fold "
                         "and fall back to 1-SE selection when it does not; 'off' is the "
                         "ablation of that capability, not a speed switch")
    ap.add_argument('--selection_rule', choices=['incumbent', 'one_se_of_best'],
                    default='incumbent',
                    help='incumbent = a challenger must clear the holder by --promotion_se standard '
                         'errors; one_se_of_best = the classic 1-SE rule over the whole pool')
    ap.add_argument('--review_mode', choices=['roles', 'single_agent', 'random'], default='roles',
                    help='roles = three participants and a chair; single_agent = one reviewer '
                         'holding all the evidence; random = a uniform draw widens a space each '
                         'round, no review (memory must be off)')
    ap.add_argument('--review_accept_policy', choices=['stop', 'require_revision'], default='stop',
                    help='stop = an ACCEPT ends the revision loop; require_revision = the fixed-horizon '
                         'RSI setting, where an ACCEPT while budget remains is answered with one demand '
                         'for the most promising revision and never ends the loop')
    ap.add_argument('--revision_reserve', type=float, default=None,
                    help='evaluation budget held back for the rounds after a review, as a fraction '
                         'of --max_evaluations and ADDED to it')
    ap.add_argument('--research_memory', choices=['off', 'evolving'], default='off',
                    help='evolving distills and revises run-local experience after each review round')
    ap.add_argument('--reserve_schedule', choices=['front', 'equal'], default='front',
                    help='front = the whole revision reserve is unlocked at the first review; equal = '
                         'round r may use up to reserve * r / retrospective, so every round can act')
    ap.add_argument('--require_llm', action='store_true')
    ap.add_argument('--selection_metric', default='official_composite')
    ap.add_argument('--reference_candidates', default=None,
                    help='explicit recipe/config pairs to re-evaluate first in development')
    ap.add_argument('--exclude_sources', default='')
    ap.add_argument('--decoder_strategies', default='plain',
                    help='development candidates: comma-separated plain,subject_consensus,metric_bias,metric_bias_consensus')
    a = ap.parse_args()

    if a.research_memory == 'evolving' and (a.evaluation != 'development' or a.retrospective < 1):
        ap.error('evolving research memory requires development evaluation and retrospective >= 1')
    if a.evaluation == 'development':
        if a.test_root is not None or not a.output_dir:
            ap.error('development requires a fresh --output_dir and forbids --test_root')
        if (a.seed < 0 or not 2 <= a.development_folds <= 20 or
                not 1 <= a.max_evaluations <= 256 or not 0 <= a.review_checks <= 20 or
                not 0 <= a.retrospective <= 20):
            ap.error('invalid development seed, folds or search/review budget')
        from merid.benchmarks.mpdd2025.development_entry import run
        run(a)
        return
    if not a.test_root:
        ap.error('--test_root is required for final evaluation')

    proposer = ClaudeProposer(model=a.llm_model, provider=a.proposer, base_url=a.llm_base_url,
                              api_key=a.llm_api_key)
    agent = AdapterAgent(proposer)
    traj = Trajectory(model=a.llm_model, config={"task": "unified method run"},
                      stream_path="method_trajectory.jsonl")
    traj.add_system_prompt("adapter_codegen", ADAPTER_SYSTEM)
    proposer.traj = traj

    rows, records = [], {}
    for track, test_sub, mm, tasks in CONFIGS:
        train_root = os.path.join(a.data_root, track, "Training")
        for window in WINDOWS:
            ctx = f"{track}/{window}"
            print(f"\n########## {ctx} ##########", flush=True)

            report = inspect_dataset({"train_root": train_root, "test_root": a.test_root})
            global SEL_METRIC
            SEL_METRIC = propose_eval_metric(
                proposer, ctx,
                ("per-segment depression classification (%s) on %s, scored per task and "
                 "averaged into a track score") % ("/".join(t for t, _ in tasks), ctx),
                report, a.freeze_dir, traj,
                corpus=os.path.basename(os.path.normpath(a.data_root)))
            print("  [eval-metric] selection will optimise %s" % SEL_METRIC, flush=True)
            mods = "audio+video sequences (opensmile/wav2vec/mfcc; openface/densenet/resnet), per-segment"
            profile_txt = f"{ctx}: multimodal {mods}; small-n per-subject; classification {[t for t,_ in tasks]}"

            method, reasoning = select_method(profile_txt + "\n\n" + report[:1500], proposer)
            traj.log("method_select", prompt=profile_txt, reasoning=reasoning, choice=method, dataset=ctx)
            print(f"  [SELECT] -> {method}", flush=True)

            if method in ("mult", "flex_moe", "CREATE"):
                print(f"  [route] SELECT chose '{method}', but it is NOT executed: no deep runner / "
                      f"method-level codegen is implemented. Falling back to the classical path. "
                      f"This is hard-coded, not an agent or OOF decision.", flush=True)
                traj.log("route_fallback", dataset=ctx, choice="classical", executed_selection=False,
                         reasoning=(f"HARD-CODED fallback: '{method}' has no runner in this codebase, so it "
                                    f"was not executed and no OOF comparison against it was performed. The "
                                    f"reported numbers come from the classical path regardless of SELECT."))
                method = "sklearn_svc"

            spec = spec_for(track, window, mm, tasks)
            load, code, data, adapter_reasoning = agent.create_adapter(
                {"train_root": train_root, "test_root": a.test_root}, spec, max_iters=5,
                freeze_path=os.path.join(a.freeze_dir, f"adapter_{ctx.replace('/', '_')}.py"))
            traj.log("adapter_codegen", reasoning=adapter_reasoning, dataset=ctx, ok=load is not None, code=code)
            if load is None:
                print(f"  [{ctx}] adapter FAILED"); continue

            recipes = propose_recipe_space(proposer, report, ctx, a.freeze_dir, traj)
            picked = select_recipe(load, recipes, tasks, traj, ctx)
            if picked is None:
                print(f"  [{ctx}] no working recipe"); continue
            best_score, best_recipe, data = picked
            print(f"  [RECIPE PICKED] {best_recipe} robust-official={best_score:.4f}", flush=True)
            import hashlib as _hl
            records[ctx] = {"method_select": method, "recipe": best_recipe,
                            "recipe_robust_official": round(float(best_score), 4),
                            "p": int(np.asarray(data["train_X"]).shape[1]),
                            "n_candidates": len(recipes),
                            "adapter_sha256": _hl.sha256((code or "").encode()).hexdigest(),
                            "code": code}

            groups = data.get("train_subject")
            ng = len(np.unique(groups)) if groups is not None else 0
            import collections
            dists = {t: dict(sorted(collections.Counter(np.asarray(data[f"train_{t}"], int).tolist()).items()))
                     for t, _ in tasks if f"train_{t}" in data}
            prof = (f"{ctx}: n_train={np.asarray(data['train_X']).shape[0]} segs from {ng} subjects, "
                    f"p={np.asarray(data['train_X']).shape[1]}; dists {dists}")
            traj.log("regime_profile", dataset=ctx, profile=prof, n_subjects=int(ng))
            print(f"  REGIME: {prof}", flush=True)
            mod_priors = propose_modality_priors(proposer, report, prof, best_recipe, ctx,
                                                 a.freeze_dir, traj)
            for _task, _lk in tasks:
                _trk = f"train_{_task}"
                if _trk not in data:
                    continue
                _y = np.asarray(data[_trk], int)
                if len(set(_y.tolist())) < 2:
                    continue
                mc = modality_credibility(load, best_recipe, _y, groups, priors=mod_priors)
                if not mc:
                    continue
                nullv = mc.get("null", {})
                print(f"  [modality-signal {_task}] full-OOF={mc['_full_oof']:.4f} "
                      f"task-null={nullv.get('verdict')}"
                      + (f" z={nullv.get('z')}" if nullv.get("z") is not None else "")
                      + f"  (permutation {'ran' if mc['null_triage']['ran'] else 'skipped'}: "
                        f"{mc['null_triage']['reason']})", flush=True)
                for _src, _v in mc["modalities"].items():
                    pw = _v.get("prior_weight")
                    print(f"      {_src:<8} marginal={_v['marginal']} -> {_v['verdict']}"
                          + (f"   (agent prior {pw}, {_v.get('prior_conf')})" if pw is not None else ""),
                          flush=True)
                cal = mc.get("calibration", {})
                if cal.get("n_tested"):
                    print(f"      prior-vs-measured: {cal['n_agree']}/{cal['n_tested']} agree "
                          f"(acc={cal['accuracy']})", flush=True)
                traj.log("modality_signal", dataset=ctx, task=_task, observation=mc)

            clf_configs = propose_clf_space(proposer, prof, ctx, a.freeze_dir, traj)
            codegen_build, codegen_code = propose_codegen_method(proposer, prof, ctx, a.freeze_dir, traj)
            records[ctx]["n_clf_candidates"] = len(clf_configs) + (1 if codegen_build else 0)
            records[ctx]["codegen_method"] = codegen_code
            classical_route(data, tasks, proposer, prof, ctx, traj, rows, track, window, clf_configs,
                            codegen_build=codegen_build)

    traj.close("method_trajectory.json")
    ms = traj.info["model_stats"]
    print(f"\n[trajectory] {ms['api_calls']} LLM calls, ${ms['cost']:.3f} -> method_trajectory.json")
    write_bundle(a, rows, records)
    print("\n==================== MERID — MPDD-2025 ====================")
    print("(WAcc = balanced accuracy; TaskAcc=(Acc+WAcc)/2; TaskF1=(MacroF1+WF1)/2; Official=(TaskAcc+TaskF1)/2)")
    print(f"{'Track':<13}{'Win':<4}{'Task':<9}{'N':>5}{'Acc':>8}{'MacroF1':>9}{'WAcc':>8}{'WF1':>8}"
          f"{'TaskAcc':>9}{'TaskF1':>8}{'Official':>9}")
    for tr, w, tk, n, acc, wacc, mf1, wf1, off in rows:
        taskacc, taskf1 = (acc + wacc) / 2, (mf1 + wf1) / 2
        print(f"{tr:<13}{w:<4}{tk:<9}{n:>5}{acc:>8.4f}{mf1:>9.4f}{wacc:>8.4f}{wf1:>8.4f}"
              f"{taskacc:>9.4f}{taskf1:>8.4f}{off:>9.4f}")


if __name__ == "__main__":
    main()
