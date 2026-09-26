"""Entry point for E-DAIC, Mental Health and Kaggle: adapter, joint search, review and final evaluation."""
import argparse
import os
import re
import sys

import numpy as np

from merid.core.grounded_state_construction.adapter_agent import (
    ADAPTER_SYSTEM, AdapterAgent, inspect_dataset)
from merid.llm.proposer import ClaudeProposer
from merid.utils.progress import stage as _stage
from merid.run_method import (
    metric_value, propose_clf_space, propose_codegen_method, propose_eval_metric,
    propose_modality_priors)
from merid.benchmarks.mpdd2025.run_all import official
from merid.utils.trajectory import Trajectory

CTX = "E-DAIC"


def corpus_spec(ctx, extractors, recipe_contract="", task=""):
    ext = ("The inspection lists these extractor directories: %s. Use those names verbatim.\n" % extractors
           if extractors else
           "The inspection lists NO directories of extracted features. If it shows raw recordings "
           "(.wav/.mp4/.mat and the like) then this corpus ships RECORDINGS, and computing features is "
           "part of your job: use librosa / soundfile / scipy.io / cv2 (import them inside load()). "
           "Name each representation you implement, and make the NAME a legal value of the matching "
           "recipe key, so the outer search can choose between them.\n")
    return (
        "Corpus %s. Build a per-example feature matrix and its labels from what the inspection shows.\n"
        % ctx
        + ext +
        "THE `recipe` ARGUMENT — you do NOT choose the recipe. An OUTER loop calls load() repeatedly "
        "with different recipe dicts and picks the winner by a leak-free score. Implement the recipe "
        "given, FAITHFULLY and DETERMINISTICALLY: the same recipe must return byte-identical features "
        "every call. A recipe maps SOURCE NAME -> the choice for that source (a string), or None to "
        "drop that source entirely, plus `agg` naming how a variable-length series is reduced to one "
        "fixed-length vector per example: 'meanstd' | 'meanstd_minmax' | 'meanstd_p25p75' | "
        "'meanstd_delta' (mean, std, and the mean absolute frame-to-frame difference). Honour `agg` for "
        "every series-valued source. If a recipe names a source you did not implement, raise - do not "
        "silently return the same features, because a search whose knobs do nothing reports a ranking "
        "of noise.\n"
        + ("THIS RUN PREDICTS `%s`, so return it as train_%s. If this corpus genuinely does not "
            "define such a label, do NOT invent one and do NOT quietly answer a different question "
            "instead: return the label(s) it really has and let the caller see the mismatch.\n"
            % (task, task) if task else "")
        + "WHICH FILES ARE THIS CORPUS. A root can hold more than one unrelated collection - a "
        "table describing one population sitting beside recordings of another, bundled into the "
        "same archive by whoever packaged it. Those are not one multimodal dataset unless "
        "something LINKS them: a shared id, a filename column, counts that line up. Check before "
        "assuming, and state at the top of load(), in a comment, which files you used and why - "
        "if you rejected a source because nothing joins it to the rest, say so. Deciding between "
        "unrelated collections is yours to do; doing it silently is not, because a reader of the "
        "result cannot tell which dataset the number describes. If the recipes below name a source "
        "you have determined cannot be honoured - nothing joins it to the labelled examples, the "
        "files will not open, the counts do not line up - then raise NotImplementedError naming that "
        "source and saying why. That is a SUPPORTED answer, not a failure: the recipes using it are "
        "dropped and the run continues. The recipe space was designed from a file listing, and you "
        "are the first thing here that actually reads the data.\n"
        "LABELS. They may be in a json or csv, or encoded in a DIRECTORY NAME or a FILENAME; read "
        "whichever this corpus uses and say which in a comment. Return one int vector per "
        "classification task and one float vector per continuous target, named train_<task> / "
        "test_<task> (and dev_<task> only if the corpus itself publishes a dev split). %s\n"
        + (recipe_contract or "") +
        "GROUPING - this one decides whether the score means anything. Return train_subject: the unit "
        "that must never be split across cross-validation folds, i.e. the person (participant, speaker, "
        "actor) a row came from, read from the id or the filename. Also return train_ids and, "
        "when present, dev_ids/test_ids: unique stable ROW identities (e.g. participant + session + "
        "segment), independent of recipe and row order. Return dev_subject/test_subject separately "
        "when a participant has multiple rows; row IDs must not substitute for grouping. "
        "Where a corpus's positive and "
        "negative examples come from different people, a model scored without grouping is scored on "
        "telling those people apart rather than on the label, and it will look excellent for the wrong "
        "reason. If every row is a distinct person, train_subject is just that person's id.\n"
        "LEAKAGE. Anything FITTED - a vocabulary, idf weights, a scaler, PCA/SVD, a codebook, a "
        "normalisation constant - is fitted on the TRAINING rows only and merely APPLIED to dev and "
        "test. Statistics computed over all examples pull the held-out data into the representation "
        "and inflate exactly the numbers this run exists to measure.\n"
        "RETURN a dict: train_X (float [n,p]), one train_<task> label vector per task, and "
        "train_subject (len n). "
        "ONLY IF THIS CORPUS PUBLISHES ITS OWN HELD-OUT SPLITS may you add test_X / test_<task> / "
        "test_ids and dev_X / dev_<task> / dev_ids. Where it does not, return every example as train "
        "and omit those blocks entirely: the runner then cross-validates over subjects and reports "
        "that. DO NOT invent a split, and above all do not return the same rows as both train and "
        "test - that is not a held-out set, it is the training data measured twice, and it turns the "
        "whole report into a training score without saying so. "
        "HOW OFTEN load() IS CALLED, because it changes what a sensible implementation looks like: "
        "an outer search calls it once per candidate recipe, dozens of times, and the recipes overlap "
        "heavily - the same source appears in many of them. Extract each source ONCE and reuse it: "
        "keep a bounded module-level dict keyed by (source name, its settings), or use the explicitly "
        "configured project cache directory in XDG_CACHE_HOME. Dataset directories are read-only; "
        "if no project cache directory is configured, use memory only. Re-decoding gigabytes of recordings for every "
        "recipe is the difference between a run that finishes and one that does not; key the cache so "
        "that two different settings can never collide. "
        "Use numpy and the standard library; import any decoder you need inside load()."
    )


AGGS = ("meanstd", "meanstd_minmax", "meanstd_p25p75", "meanstd_delta")

RECIPE_SYSTEM = (
    "You DESIGN the candidate feature-recipe SPACE for a SMALL-SAMPLE (p>>n) clinical multimodal task. "
    "Reason from what the inspection shows ACTUALLY EXISTS on this disk, then output ONLY a JSON list "
    "of candidate recipes.\n"
    "A recipe is a flat dict. Each key is a SOURCE you will ask the loader for; its value is the "
    "choice for that source, or null to drop it. One reserved key: `agg`, one of {aggs}, saying how a "
    "variable-length series becomes one fixed-length vector.\n"
    "{menu}"
    "WHERE THE NAMES COME FROM. If the corpus ships extracted features, the source names are the "
    "extractor directories, verbatim - do not invent names and do not carry names over from another "
    "corpus. If it ships RAW RECORDINGS, there are no directories to read, so you name the "
    "representations yourself (for example an acoustic key whose values are different MFCC or "
    "log-mel settings); the loader will be told to implement exactly the names you use here, so keep "
    "them descriptive and stable.\n"
    "WHAT A GOOD SPACE COVERS. The point is to span the plausible range, not to bet on one end of it. "
    "Include each source ALONE (that is how the search learns which one carries the signal, and on "
    "the corpora seen so far the answer has differed between tasks on the SAME data), the "
    "combinations worth trying, both a high-dimensional and a low-dimensional option where the corpus "
    "offers both - at small n a several-thousand-column source often contributes variance rather than "
    "signal, and a twenty-column one can be too coarse - and at least one temporal-dynamics "
    "(meanstd_delta) candidate. "
    "SOURCES THAT DEFINE THE LABEL RATHER THAN PREDICT IT: some candidates are not evidence about the "
    "outcome, they ARE the instrument the outcome was assigned with - a questionnaire the diagnosis "
    "was made from, a score the classes were thresholded out of. Include one and the search returns a "
    "near-perfect number that measures the label's own definition rather than any ability to detect "
    "it. Say which of your sources you believe sit UPSTREAM of the label and why, and make sure the "
    "space still contains recipes that answer the question WITHOUT them, so both readings are "
    "available. "
    "Where the sources are COLUMN GROUPS of one table, cover every column "
    "between them and include one recipe with all groups on: a group left out of the space can never "
    "be found, and on one corpus the omitted group was the one carrying the signal. "
    "Include 8-14 DISTINCT recipes; a leak-free search picks the winner, "
    "so propose a good diverse SPACE rather than one guess. Output ONLY the JSON list."
)


def _parse_recipes(txt, known=()):
    import json as _json
    import re as _re
    spans = _re.findall(r"```(?:json)?\s*(\[.*?\])\s*```", txt or "", _re.S)
    spans += [(txt or "")[i:] for i in
              sorted((m.start() for m in _re.finditer(r"\[", txt or "")), reverse=True)]
    dec = _json.JSONDecoder()
    raw = None
    for sp in spans:
        try:
            obj, _ = dec.raw_decode(sp.strip())
        except Exception:
            continue
        if isinstance(obj, list) and any(isinstance(x, dict) for x in obj):
            raw = obj
            break
    if raw is None:
        return [], ["no JSON list of recipes in the answer"]
    out, errs, seen = [], [], set()
    for r in raw:
        if not isinstance(r, dict):
            continue
        rec, bad = {}, []
        for k, v in r.items():
            if k == "agg":
                continue
            if v in (None, "", "none", "null", False):
                rec[k] = None
            elif isinstance(v, str):
                rec[k] = v
                if known and v not in known:
                    errs.append("%s=%r is not one of the discovered extractors %s "
                                "(fine if this corpus ships recordings and you are naming your own "
                                "representations)" % (k, v, list(known)[:6]))
            else:
                bad.append("%s=%r is neither a string nor null" % (k, v))
        if not any(rec.values()):
            bad.append("no source enabled")
        if bad:
            errs.append("%s -> %s" % (r, "; ".join(bad)))
            continue
        rec["agg"] = r.get("agg") if r.get("agg") in AGGS else "meanstd"
        key = _json.dumps(rec, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(rec)
    return out, errs


def _default_space(extractors):
    ex = list(extractors)
    if not ex:
        return []
    sp = []
    for agg in ("meanstd", "meanstd_minmax"):
        for e in ex:
            sp.append({"src": e, "agg": agg})
    for i, e in enumerate(ex[:3]):
        for f in ex[i + 1:3]:
            sp.append({"src": e, "src2": f, "agg": "meanstd"})
    return sp


REPORT_CHARS = int(os.environ.get("MERID_REPORT_CHARS", "20000"))


def propose_recipe_space(proposer, report, extractors, freeze_dir, traj=None, feedback=""):
    import hashlib
    import json as _json
    fp = os.path.join(freeze_dir, "recipe_space_%s.json" % CTX.replace("/", "_"))
    if os.path.exists(fp):
        obj = _json.load(open(fp))
        print("  [recipe-space] reused %d FROZEN recipes (%s)" % (len(obj["recipes"]), fp), flush=True)
        return obj["recipes"]
    from merid.llm.retrieval import decide_with_optional_search
    menu = ("The inspection lists these extractor directories, and they ARE the legal source values: "
            "%s\n" % list(extractors) if extractors else
            "The inspection lists no extractor directories. If it shows raw recordings, name the "
            "representations you want the loader to implement and vary them across recipes.\n")
    system = RECIPE_SYSTEM.format(aggs=" | ".join("'%s'" % a for a in AGGS), menu=menu)
    NL = chr(10)
    prompt = ("Corpus: %s" + NL + NL + "OBSERVED INSPECTION (what exists, with shapes and value "
              "statistics):" + NL + "%s" + NL + NL + "%sDesign the candidate recipe list now."
              ) % (CTX, report[:REPORT_CHARS], feedback)
    txt = decide_with_optional_search(proposer, system, prompt, traj=traj, ctx=CTX, tag="recipe-space")
    recipes, errs = _parse_recipes(txt, extractors)
    for e in errs[:6]:
        print("  [recipe-space] NOTE %s" % e, flush=True)
    if not recipes:
        recipes = _default_space(extractors)
        if recipes:
            print("  [recipe-space] no usable proposal -> fallback over discovered extractors (%d)"
                  % len(recipes), flush=True)
        else:
            raise SystemExit(
                "[%s] the agent proposed no usable recipe and there are no extractor directories to "
                "fall back on. This corpus ships recordings, so the recipe space can only come from "
                "the agent; re-run to ask again rather than substituting a guess of ours." % CTX)
    os.makedirs(freeze_dir, exist_ok=True)
    payload = {"ctx": CTX, "recipes": recipes, "reasoning": (txt or "")[:3000],
               "discovered_extractors": list(extractors)}
    payload["sha256"] = hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()
    with open(fp, "w") as fh:
        _json.dump(payload, fh, indent=2)
    srcs = sorted({k for r in recipes for k, v in r.items() if k != "agg" and v})
    print("  [recipe-space] agent proposed %d recipes over sources %s -> FROZEN"
          % (len(recipes), srcs), flush=True)
    if traj is not None:
        traj.log("recipe_space_propose", dataset=CTX, choice=recipes,
                 reasoning=(txt or "")[:3000], sha256=payload["sha256"])
    return recipes


TASK = "binary"

REG_TASKS = ("phq", "score", "severity", "total")


def _ykey():
    return TASK


def _is_reg():
    return any(t in TASK.lower() for t in REG_TASKS)


def _y(d, split):
    return np.asarray(d["%s_%s" % (split, _ykey())], float if _is_reg() else int)


def _make(cfg, p):
    from merid.benchmarks.mpdd2025.run_all import make_clf, make_reg
    return (make_reg if _is_reg() else make_clf)(cfg, p)


def _probe_cfg():
    from merid.run_method import PROBE_CFG
    return {"model": "ridge", "k": 128, "alpha": 100.0} if _is_reg() else PROBE_CFG


SEL_NAME = "macro_f1"


def _dev(yd, yp):
    return metric_value(SEL_NAME, yd, yp)


def _strata(y):
    y = np.asarray(y)
    if not _is_reg():
        return y.astype(int)
    q = np.quantile(y, [0.25, 0.5, 0.75])
    return np.digitize(y, q)


def _has_dev(d):
    return "dev_X" in d and ("dev_%s" % _ykey()) in d


def _held_out_score(d, cfg, codegen_build=None, want_pred=False):
    X = np.asarray(d["train_X"], float)
    y = _y(d, "train")
    def _fresh():
        return codegen_build() if cfg == "codegen" else _make(cfg, X.shape[1])
    if _has_dev(d):
        m = _fresh()
        m.fit(X, y)
        yd = _y(d, "dev")
        pr = m.predict(np.asarray(d["dev_X"], float))
        sc = _dev(yd, pr)
        return (sc, yd, pr, None) if want_pred else sc
    from merid.benchmarks.mpdd2025.run_all import grouped_folds
    g = np.asarray(d.get("train_subject", np.arange(len(y))))
    oof = np.full(len(y), np.nan)
    for tr, va in grouped_folds(_strata(y), g, seed=0):
        if not _is_reg() and len(set(y[tr].tolist())) < 2:
            continue
        m = _fresh()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    mk = ~np.isnan(oof)
    if mk.sum() < max(4, 0.5 * len(y)):
        return (None, None, None, None) if want_pred else None
    pr = oof[mk] if _is_reg() else oof[mk].astype(int)
    sc = _dev(y[mk], pr)
    return (sc, y[mk], pr, g[mk]) if want_pred else sc


def _score_spread(d, cfg, codegen_build=None):
    sc, yt, pr, g = _held_out_score(d, cfg, codegen_build, want_pred=True)
    if sc is None:
        return 0.0
    if g is None:
        return _dev_score_spread(np.asarray(yt), np.asarray(pr))
    from merid.run_method import _subject_bootstrap_se
    return _subject_bootstrap_se(yt, pr, g, SEL_NAME)


def _fit_predict(cfg, codegen_build, X, y, Xq):
    m = codegen_build() if cfg == "codegen" else _make(cfg, X.shape[1])
    m.fit(X, y)
    return np.asarray(m.predict(Xq), float if _is_reg() else int)


def select_recipe(load, recipes, traj):
    best, venue = None, None
    for rc in recipes:
        try:
            d = load(rc)
        except Exception as exc:
            print(f"  [recipe {rc}] load FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            continue
        try:
            X = np.asarray(d["train_X"], float)
            y = _y(d, "train")
        except KeyError as exc:
            print(f"  [recipe {rc}] adapter returned no {exc}", flush=True)
            continue
        if X.ndim != 2 or X.shape[0] == 0 or (not _is_reg() and len(set(y.tolist())) < 2):
            continue
        if venue is None:
            venue = "dev" if _has_dev(d) else "grouped-OOF"
            print(f"  [venue] selection scores on {venue}"
                  + ("" if venue == "dev" else " inside train (this corpus publishes no dev split)"),
                  flush=True)
        try:
            sc = _held_out_score(d, _probe_cfg())
        except Exception as exc:
            print(f"  [recipe {rc}] p={X.shape[1]} UNUSABLE -> {type(exc).__name__}: "
                  f"{str(exc)[:110]}", flush=True)
            if traj is not None:
                traj.log("recipe_select", dataset=CTX, choice=rc, p=int(X.shape[1]), valid=False,
                         error=f"{type(exc).__name__}: {str(exc)[:200]}")
            continue
        if sc is None:
            print(f"  [recipe {rc}] p={X.shape[1]} UNUSABLE -> too few rows covered by the folds",
                  flush=True)
            continue
        print(f"  [recipe] {rc} -> p={X.shape[1]} {venue}-{SEL_NAME}={sc:.4f}", flush=True)
        if traj is not None:
            traj.log("recipe_select", dataset=CTX, choice=rc, p=int(X.shape[1]), venue=venue,
                     score=round(float(sc), 4))
        if best is None or sc > best[0]:
            best = (float(sc), rc, d)
    return best


def _complexity(cfg):
    if cfg == "codegen":
        return (2, 10 ** 6, 9)
    fam = {"svc_linear": 0, "logreg": 0, "ridge": 0, "svr_linear": 0,
           "svc_rbf": 1, "svr_rbf": 1, "gb": 2, "gbr": 2, "rf": 3, "rfr": 3, "et": 3, "etr": 3}
    return (0 if isinstance(cfg, dict) else 1, int(cfg.get("k", 128)) if isinstance(cfg, dict) else 10 ** 6,
            fam.get(cfg.get("model") if isinstance(cfg, dict) else "", 5))


def _dev_score_spread(yd, preds, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(yd)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(yd[idx].tolist())) < 2:
            continue
        out.append(_dev(yd[idx], preds[idx]))
    return float(np.std(out)) if out else 0.0


def select_model(d, Xt, configs, codegen_build, traj):
    X = np.asarray(d["train_X"], float)
    y = _y(d, "train")
    venue = "dev" if _has_dev(d) else "grouped-OOF"
    cands = list(configs) + (["codegen"] if codegen_build is not None else [])
    scored, errs = [], []
    for cfg in cands:
        try:
            sc = _held_out_score(d, cfg, codegen_build)
        except Exception as exc:
            print(f"    [clf] {cfg} -> FAILED {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            errs.append(f"{type(exc).__name__}: {exc}")
            continue
        if sc is None:
            continue
        star = "  <-- CODEGEN" if cfg == "codegen" else ""
        print(f"    [clf] {cfg} -> {venue}-{SEL_NAME}={sc:.4f}{star}", flush=True)
        if traj is not None:
            traj.log("clf_select", dataset=CTX, task=TASK, choice=str(cfg), venue=venue,
                     score=round(float(sc), 4), codegen=(cfg == "codegen"))
        scored.append((float(sc), cfg))
    if not scored:
        raise RuntimeError(
            f"[{CTX}] every model candidate failed — this is a BUG, not a result. "
            f"First cause: {next((e for e in errs if e), 'unknown')}")
    scored.sort(key=lambda z: z[0], reverse=True)
    top_score, top_cfg = scored[0]
    sd = _score_spread(d, top_cfg, codegen_build)
    tied = sorted([(sc, c) for sc, c in scored if sc >= top_score - sd],
                  key=lambda z: _complexity(z[1]))
    score, cfg = tied[0]
    print(f"    [1-SE] {venue} bootstrap sd={sd:.4f}; {len(tied)} candidate(s) within 1 SE of "
          f"{top_score:.4f} -> keeping the simplest", flush=True)
    if cfg != top_cfg:
        print(f"    [1-SE] top was {top_cfg} ({top_score:.4f}); its lead of "
              f"{top_score - score:.4f} does not clear the noise -> {cfg} ({score:.4f})", flush=True)
    if traj is not None:
        traj.log("clf_select_1se", dataset=CTX, task=TASK, choice=str(cfg), venue=venue,
                 top=str(top_cfg), top_score=round(top_score, 4), chosen=round(float(score), 4),
                 boot_sd=round(sd, 4), n_tied=len(tied))
    return cfg, score, _fit_predict(cfg, codegen_build, X, y, Xt)


def modality_credibility(load, recipe, priors, traj):
    priors = priors or {}

    def score(rc):
        try:
            d = load(rc)
            X = np.asarray(d["train_X"], float)
        except Exception:
            return None
        if X.ndim != 2 or X.shape[1] == 0:
            return None
        try:
            return _held_out_score(d, _probe_cfg())
        except Exception:
            return None

    full = score(recipe)
    if full is None:
        return None
    try:
        FLOOR = _score_spread(load(recipe), _probe_cfg())
    except Exception:
        FLOOR = 0.05
    print(f"  [modality-signal] floor = bootstrap sd of {SEL_NAME} = {FLOOR:.4f}", flush=True)
    srcs = [k for k, v in recipe.items() if k != "agg" and v]
    out = {"full": round(full, 4), "modalities": {}, "n_tested": 0, "n_agree": 0, "floor": FLOOR}
    for k in srcs:
        ab = dict(recipe)
        ab[k] = None
        loo = score(ab)
        if loo is None and not any(v for j, v in ab.items() if j != "agg"):
            try:
                d1 = load(recipe)
                ytr = _y(d1, "train")
                if _has_dev(d1):
                    tgt = _y(d1, "dev")
                else:
                    tgt = ytr
                if _is_reg():
                    base = np.full(len(tgt), float(np.mean(ytr)))
                else:
                    vals, cnt = np.unique(ytr, return_counts=True)
                    base = np.full(len(tgt), vals[int(np.argmax(cnt))])
                loo = _dev(tgt, base)
                print(f"      ({k} is the only source; comparing against no features at all — "
                      f"the trivial predictor scores {loo:.4f})", flush=True)
            except Exception:
                loo = None
        pri = priors.get(k, {})
        if loo is None:
            out["modalities"][k] = {"loo": None, "marginal": 0.0, "verdict": "UNTESTABLE",
                                    "prior_weight": pri.get("weight"),
                                    "prior_conf": pri.get("confidence")}
            continue
        marg = full - loo
        verdict = ("SIGNAL" if marg > FLOOR else "NOISE" if marg < -FLOOR else "INCONCLUSIVE")
        out["modalities"][k] = {"loo": round(loo, 4), "marginal": round(marg, 4), "verdict": verdict,
                                "prior_weight": pri.get("weight"),
                                "prior_conf": pri.get("confidence")}
        if pri.get("weight") is not None:
            out["n_tested"] += 1
            said_matters = float(pri["weight"]) >= 0.34
            out["n_agree"] += int(said_matters == (verdict == "SIGNAL"))
    if traj is not None:
        traj.log("modality_ablation", dataset=CTX, task=TASK, observation=out)
    return out


def discover_extractors(root):
    for d in sorted(os.listdir(root)):
        fdir = os.path.join(root, d, "features")
        if not os.path.isdir(fdir):
            continue
        pid = d.split("_")[0]
        names = []
        for f in sorted(os.listdir(fdir)):
            if f.lower().endswith(".csv"):
                names.append(f[len(pid) + 1:-4] if f.startswith(pid + "_") else f[:-4])
        if names:
            return names
    return []


def _oof_pred(X, y, g, cfg, codegen_build=None, seed=0):
    from merid.benchmarks.mpdd2025.run_all import grouped_folds
    oof = np.full(len(y), np.nan)
    for tr, va in grouped_folds(_strata(y), g, seed=seed):
        if len(tr) == 0 or len(va) == 0:
            continue
        if not _is_reg() and len(set(y[tr].tolist())) < 2:
            continue
        m = codegen_build() if cfg == "codegen" else _make(cfg, X.shape[1])
        m.fit(X[tr], y[tr])
        oof[va] = m.predict(X[va])
    return oof


def nested_report(load, recipes, configs, codegen_build, traj, widths=None):
    from merid.benchmarks.mpdd2025.run_all import grouped_folds
    import json as _json
    mats = {}
    for rc in recipes:
        try:
            d = load(rc)
            X = np.asarray(d["train_X"], float)
            y = _y(d, "train")
            g = np.asarray(d.get("train_subject", np.arange(len(y))))
        except NotImplementedError as exc:
            print("  [recipe %s] VETOED by the adapter: %s" % (rc, str(exc)[:170]), flush=True)
            continue
        except Exception as exc:
            print("  [recipe %s] load FAILED: %s: %s" % (rc, type(exc).__name__, str(exc)[:110]),
                  flush=True)
            continue
        if X.ndim != 2 or X.shape[1] == 0:
            print("  [recipe %s] UNUSABLE -> %d columns" % (rc, 0 if X.ndim < 2 else X.shape[1]),
                  flush=True)
            continue
        keep = np.nanstd(X, axis=0) > 0
        if not keep.any():
            print("  [recipe %s] UNUSABLE -> all %d columns are constant (the source produced no "
                  "variation across examples)" % (rc, X.shape[1]), flush=True)
            continue
        if not keep.all():
            print("  [recipe %s] %d of %d columns are constant and will be dropped by the pipeline"
                  % (rc, int((~keep).sum()), X.shape[1]), flush=True)
        if widths is not None:
            widths[str(rc)] = {"p": int(X.shape[1]), "constant_columns": int((~keep).sum())}
        mats[_json.dumps(rc, sort_keys=True)] = (rc, X, y, g)
    if not mats:
        return None
    _rc0, _X0, y, g = next(iter(mats.values()))
    n_sub = len(np.unique(g))
    if n_sub < 8:
        print("  [nested] ONLY %d grouping units (%s) — a subject-disjoint nested CV needs enough "
              "groups to split twice, so this corpus cannot be evaluated that way. Reporting nothing "
              "rather than a number produced by folds that overlap or lose a class."
              % (n_sub, ", ".join(map(str, np.unique(g)[:8]))), flush=True)
        if traj is not None:
            traj.log("protocol_refused", dataset=CTX, task=TASK, n_groups=int(n_sub),
                     groups=[str(x) for x in np.unique(g)[:16]],
                     reason="too few grouping units for subject-disjoint nested CV")
        return None
    import collections
    dist = ("target mean=%.2f sd=%.2f range[%.0f,%.0f]" % (y.mean(), y.std(), y.min(), y.max())
            if _is_reg() else "label dist %s"
            % dict(sorted(collections.Counter(np.asarray(y, int).tolist()).items())))
    print("  REGIME: %s: n=%d, p=%d, %d groups; %s"
          % (CTX, len(y), _X0.shape[1], n_sub, dist), flush=True)
    if not _is_reg() and len(np.unique(np.asarray(y, int))) < 2:
        print("  [nested] the '%s' label has ONE class — there is nothing to predict. The adapter "
              "returned a constant label vector, which usually means this corpus does not define the "
              "task that was asked for." % TASK, flush=True)
        if traj is not None:
            traj.log("protocol_refused", dataset=CTX, task=TASK, reason="single-class label")
        return None
    outer = grouped_folds(_strata(y), g, seed=0)
    print("  [nested] %d subject-disjoint outer folds over %d groups; selection repeated inside each"
          % (len(outer), n_sub), flush=True)
    pred = np.full(len(y), np.nan)
    picks = []
    for i, (tr, te) in enumerate(outer):
        best = None
        for key, (rc, X, _y2, _g) in mats.items():
            try:
                sc_oof = _oof_pred(X[tr], y[tr], g[tr], _probe_cfg())
            except Exception as exc:
                print("    [fold %d] recipe %s dropped: %s: %s"
                      % (i + 1, rc, type(exc).__name__, str(exc)[:90]), flush=True)
                continue
            mk = ~np.isnan(sc_oof)
            if mk.sum() < 4:
                continue
            sc = _dev(y[tr][mk], sc_oof[mk] if _is_reg() else sc_oof[mk].astype(int))
            if best is None or sc > best[0]:
                best = (sc, key)
        if best is None:
            print("    [fold %d] no recipe produced a usable inner OOF — every candidate's folds "
                  "were empty or single-class" % (i + 1), flush=True)
            continue
        rc, X, _y2, _g = mats[best[1]]
        scored = []
        for cfg in list(configs) + (["codegen"] if codegen_build is not None else []):
            try:
                o = _oof_pred(X[tr], y[tr], g[tr], cfg, codegen_build)
            except Exception:
                continue
            mk = ~np.isnan(o)
            if mk.sum() < 4:
                continue
            scored.append((_dev(y[tr][mk], o[mk] if _is_reg() else o[mk].astype(int)), cfg, o, mk))
        if not scored:
            print("    [fold %d] recipe %s chosen but every model candidate failed on it"
                  % (i + 1, rc), flush=True)
            continue
        scored.sort(key=lambda z: z[0], reverse=True)
        top_sc, top_cfg, top_o, top_mk = scored[0]
        from merid.run_method import _subject_bootstrap_se
        sd = _subject_bootstrap_se(y[tr][top_mk],
                                   top_o[top_mk] if _is_reg() else top_o[top_mk].astype(int),
                                   g[tr][top_mk], SEL_NAME)
        tied = sorted([(sc, c) for sc, c, _o, _m in scored if sc >= top_sc - sd],
                      key=lambda z: _complexity(z[1]))
        sc, cfg = tied[0]
        m = codegen_build() if cfg == "codegen" else _make(cfg, X.shape[1])
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
        picks.append((rc, cfg, round(float(best[0]), 4), round(float(sc), 4), round(sd, 4), len(tied)))
        print("    [fold %d] recipe %s | model %s | inner %s=%.4f (SE %.4f, %d tied)"
              % (i + 1, rc, cfg, SEL_NAME, sc, sd, len(tied)), flush=True)
    mk = ~np.isnan(pred)
    if mk.sum() < 4:
        return None
    if traj is not None:
        traj.log("nested_cv", dataset=CTX, task=TASK, n_folds=len(outer), n_groups=int(n_sub),
                 picks=[{"recipe": r, "model": str(c), "recipe_score": a2, "model_score": b2,
                         "se": s2, "n_tied": t2} for r, c, a2, b2, s2, t2 in picks])
    return y[mk], (pred[mk] if _is_reg() else pred[mk].astype(int)), mk.sum(), picks


_REPORT_DIR = None


def _report(yt, yp, n, protocol, picks, traj, ref=None):
    from merid.run_method import metric_value as _mv
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    print("", flush=True)
    print("==================== MERID - %s / %s ====================" % (CTX, TASK), flush=True)
    print("  protocol: %s   (selection optimised %s)" % (protocol, SEL_NAME), flush=True)
    if _is_reg():
        row = lambda a2, b2: (-_mv("neg_rmse", a2, b2), -_mv("neg_mae", a2, b2),
                              _mv("ccc", a2, b2), _mv("r2", a2, b2))
        r = row(yt, yp)
        c = row(yt, np.full(len(yt), float(np.mean(yt))))
        print("  columns: RMSE | MAE | CCC | R2", flush=True)
        print("  RESULT (n=%d)  %.4f | %.4f | %+.4f | %+.4f" % (n, r[0], r[1], r[2], r[3]), flush=True)
        print("  predict-the-mean: %.4f | %.4f | +0.0000 | %+.4f" % (c[0], c[1], c[3]), flush=True)
        obs = {"rmse": round(r[0], 4), "mae": round(r[1], 4), "ccc": round(r[2], 4),
               "r2": round(r[3], 4)}
    else:
        from sklearn.metrics import cohen_kappa_score
        acc, wacc, mf1, wf1, off = official(yt, yp)
        kap = float(cohen_kappa_score(yt, yp))
        maj = int(np.bincount(np.asarray(yt, int)).argmax())
        cy = np.full(len(yt), maj, int)
        ca, cw, cm, cwf, _ = official(yt, cy)
        print("  columns: Acc | Bal-Acc | Macro-F1 | Weighted-F1 | Kappa", flush=True)
        print("  RESULT (n=%d)  %.4f | %.4f | %.4f | %.4f | %.4f" % (n, acc, wacc, mf1, wf1, kap),
              flush=True)
        print("  predict-the-majority: %.4f | %.4f | %.4f | %.4f | %.4f"
              % (ca, cw, cm, cwf, float(cohen_kappa_score(yt, cy))), flush=True)
        obs = {"acc": round(acc, 4), "bal_acc": round(wacc, 4), "macro_f1": round(mf1, 4),
               "weighted_f1": round(wf1, 4), "kappa": round(kap, 4), "composite": round(off, 4)}
    if ref:
        print(ref, flush=True)
    if picks:
        chosen = {str(c) for _r, c, *_ in picks}
        recs = {str(r) for r, _c, *_ in picks}
        print("  across folds: %d distinct recipes, %d distinct models chosen"
              % (len(recs), len(chosen)), flush=True)
    if traj is not None:
        traj.log("result", dataset=CTX, task=TASK, protocol=protocol, observation=obs)
    if _REPORT_DIR:
        import json as _j
        keep = {"neg_rmse": -obs["rmse"], "neg_mae": -obs["mae"], "ccc": obs["ccc"], "r2": obs["r2"]}             if _is_reg() else {"accuracy": obs["acc"], "balanced_accuracy": obs["bal_acc"],
                               "macro_f1": obs["macro_f1"], "weighted_f1": obs["weighted_f1"],
                               "cohen_kappa": obs["kappa"], "official": obs["composite"]}
        out = os.path.join(_REPORT_DIR, "nested_metrics.json")
        try:
            with open(out, "w", encoding="utf-8") as fh:
                _j.dump({"ctx": CTX, "task": TASK, "selection_metric": SEL_NAME, "n": int(n),
                         "protocol": protocol, "metrics": keep,
                         "n_distinct_recipes": len({str(r) for r, _c, *_ in picks}) if picks else None,
                         "n_distinct_models": len({str(c) for _r, c, *_ in picks}) if picks else None},
                        fh, indent=2)
            print("  metrics -> %s" % out, flush=True)
        except OSError as exc:
            print("  metrics not written (%s)" % exc, flush=True)


def _review_tools(load, recipes, report):
    def _idx(a):
        i = int(re.sub(r"[^0-9-]", "", str(a)) or -1)
        if not 0 <= i < len(recipes):
            raise ValueError("recipe index must be 0..%d" % (len(recipes) - 1))
        return i

    def recipe_list(_arg=""):
        return {str(i): {k: v for k, v in r.items() if v or k == "agg"}
                for i, r in enumerate(recipes)}

    def inspection(kw):
        kw = str(kw).strip().lower()
        hits = [ln for ln in report.splitlines() if kw in ln.lower()]
        return hits[:20] or ["no line of the inspection report contains %r" % kw]

    def probe_recipe(arg):
        d = load(recipes[_idx(arg)])
        X = np.asarray(d["train_X"], float)
        sd = np.nanstd(X, axis=0)
        return {"recipe": recipes[_idx(arg)], "n_rows": int(X.shape[0]), "p": int(X.shape[1]),
                "constant_columns": int((sd <= 0).sum()),
                "column_sd_min": float(np.nanmin(sd)), "column_sd_median": float(np.nanmedian(sd)),
                "column_sd_max": float(np.nanmax(sd)),
                "n_distinct_rows": int(len(np.unique(np.round(X, 9), axis=0)))}

    def label_distribution(_arg=""):
        d = load(recipes[0])
        yv = _y(d, "train")
        if _is_reg():
            return {"n": int(len(yv)), "mean": float(yv.mean()), "sd": float(yv.std()),
                    "min": float(yv.min()), "max": float(yv.max())}
        import collections as _c3
        cnt = dict(sorted(_c3.Counter(np.asarray(yv, int).tolist()).items()))
        n = int(len(yv))
        return {"task": TASK, "n": n, "counts": cnt,
                "majority_share": round(max(cnt.values()) / n, 4),
                "n_subjects": int(len(np.unique(np.asarray(
                    d.get("train_subject", np.arange(n))))))}

    def score_recipe(arg):
        d = load(recipes[_idx(arg)])
        return {"recipe": recipes[_idx(arg)], "selection_venue_score": _held_out_score(d, _probe_cfg()),
                "metric": SEL_NAME, "note": "fixed probe model, selection venue only"}

    return {
        "planner": {"inspection": ("keyword - lines of the inspection report containing it", inspection),
                    "label_distribution": ("- the class counts / target statistics actually loaded",
                                           label_distribution),
                    "recipe_list": ("- the recipes with their index", recipe_list)},
        "engineer": {"probe_recipe": ("index - load it and report rows, width, constant columns, "
                                      "column sd range, number of distinct rows", probe_recipe),
                     "recipe_list": ("- the recipes with their index", recipe_list),
                     "inspection": ("keyword - lines of the inspection report containing it", inspection)},
        "reviewer": {"score_recipe": ("index - its score on the selection venue with the fixed probe "
                                      "model", score_recipe),
                     "label_distribution": ("- the class counts / target statistics actually loaded",
                                            label_distribution),
                     "recipe_list": ("- the recipes with their index", recipe_list)},
    }


def _selection_evidence(recipes, adapter_reasoning, vetoes, widths, picks, folds_note,
                        freeze_dir, ctx, extra=None):
    import json as _json
    safe = ctx.replace("/", "_")
    frozen = {}
    for name in ("eval_metric_%s_clf" % safe, "eval_metric_%s_reg" % safe,
                 "recipe_space_%s" % safe, "clf_space_%s" % safe, "reg_space_%s" % safe,
                 "codegen_%s" % safe):
        fp = os.path.join(freeze_dir, name + ".json")
        if not os.path.exists(fp):
            continue
        try:
            obj = _json.load(open(fp, encoding="utf-8"))
        except Exception:
            continue
        frozen[name] = {k: (str(v)[:1400] if k in ("reasoning", "decision") else v)
                        for k, v in obj.items() if k != "code"}
    return {
        "planner": {"corpus": ctx, "task": TASK, "metric_optimised": SEL_NAME,
                    "frozen_decisions": frozen},
        "engineer": {"adapter_attempts": str(adapter_reasoning)[-2500:],
                     "sources_refused": vetoes,
                     "feature_width_per_recipe": widths,
                     "n_recipes_in_space": len(recipes)},
        "reviewer": dict({"selection_venue": folds_note, "per_fold_choices": picks}, **(extra or {})),
    }


def _restrict_recipe_space(recipes, exclude_sources=""):
    import json
    drop = {name.strip() for name in exclude_sources.split(",") if name.strip()}
    kept, seen = [], set()
    for recipe in recipes:
        restricted = {key: (None if key != "agg" and key in drop else value)
                      for key, value in recipe.items()}
        if not any(value for key, value in restricted.items() if key != "agg"):
            continue
        signature = json.dumps(restricted, sort_keys=True)
        if signature not in seen:
            seen.add(signature)
            kept.append(restricted)
    if not kept:
        raise ValueError("excluding %s leaves no recipe with any source enabled" % sorted(drop))
    return kept


def _contract_and_probes(recipes):
    import json as _json
    keys = sorted({k for r in recipes for k in r if k != "agg"})
    vals = {k: sorted({r[k] for r in recipes if r.get(k)}) for k in keys}
    NL = chr(10)
    contract = ("THE RECIPES YOU WILL BE CALLED WITH - this list is fixed, and load() must implement "
                "every key and every value in it:" + NL + "%s" + NL +
                "The keys are: %s. For each, the values it can take are:" + NL + "%s" + NL +
                "A key set to null means DROP that source for this call. Implement each value exactly "
                "as its name describes; where a name is a settings string (for example "
                "'n_mfcc=40,hop=10ms'), parse it and honour those settings. Two different values of "
                "the same key MUST produce different feature widths or different numbers - a loader "
                "that returns the same matrix whatever it is asked for turns the search above into a "
                "ranking of noise." + NL
                ) % (_json.dumps(recipes[:12], indent=1), keys,
                     NL.join("  %s: %s" % (k, v) for k, v in vals.items()))

    probes, seen_src = [], []
    for r in recipes:
        for k, v in r.items():
            if k == "agg" or not v:
                continue
            if k not in [j for j, _ in seen_src]:
                seen_src.append((k, v))
    for i in range(len(seen_src) - 1):
        (ka, va), (kb, vb) = seen_src[i], seen_src[i + 1]
        one = {ka: va, "agg": "meanstd"}
        two = {ka: va, kb: vb, "agg": "meanstd"}
        probes.append(("adding the source %r to a recipe that already has %r must ADD columns"
                       % (kb, ka), one, two))
        if len(probes) >= 12:
            break
    if not probes:
        for r in recipes[:3]:
            base = {k: v for k, v in r.items() if k != "agg"}
            if any(base.values()):
                probes.append(("'agg' widens from meanstd (2 stats/feature) to meanstd_minmax (4)",
                               dict(base, agg="meanstd"), dict(base, agg="meanstd_minmax")))
    return contract, probes


def _loader_contract(recipes, proposed, exclude_sources=""):
    # Source exclusion narrows the search space, and the loader stays checked against every
    # proposed recipe.
    return _contract_and_probes(proposed if exclude_sources else recipes)


def _apply_revision(verdict, proposer, report, extractors, contract, probes, agent, a, traj,
                    recipes, clf_configs, kind, nprof, safe, load_now=None, extra_spec="",
                    contract_checks=(), contract_checks_for=None):
    step, note = verdict["step"], verdict["note"]
    load = code = None
    def _supersede(name):
        fp = os.path.join(a.freeze_dir, name)
        if os.path.exists(fp):
            os.replace(fp, fp + ".superseded_by_review")
            print("  [review] superseded %s" % name, flush=True)
    fb = ("A POST-RUN REVIEW OF YOUR PREVIOUS ANSWER ASKED FOR THIS TO BE REDONE. What the review "
          "said:" + chr(10) + note + chr(10) + chr(10))
    if step == "eval_metric":
        global SEL_NAME
        _supersede("eval_metric_%s_%s.json" % (safe, "reg" if _is_reg() else "clf"))
        SEL_NAME = propose_eval_metric(
            proposer, CTX, fb + ("predicting `%s`" % TASK), report, a.freeze_dir, traj,
            kind=kind, corpus=CTX)
        print("  [review] metric is now %s" % SEL_NAME, flush=True)
    elif step == "recipe_space":
        _supersede("recipe_space_%s.json" % safe)
        recipes = propose_recipe_space(proposer, report, extractors, a.freeze_dir, traj, feedback=fb)
        contract, probes = _contract_and_probes(recipes)
        from merid.core.grounded_state_construction.adapter_agent import check_recipe_honored
        err = check_recipe_honored(load_now, probes) if (load_now and probes) else ""
        if err:
            print("  [review] the frozen adapter does not honour the revised space -> re-deriving; "
                  "%s" % err[:180], flush=True)
            fp = _adapter_freeze_path(a.freeze_dir, safe)
            if os.path.exists(fp):
                os.replace(fp, fp + ".superseded_by_review")
            load, code, _d, _r = agent.create_adapter(
                {"train_root": a.data_root, "test_root": a.data_root},
                corpus_spec(CTX, extractors, contract + chr(10) + fb, task=_ykey()) + extra_spec, max_iters=a.adapter_iters,
                freeze_path=fp, probes=probes, contract_checks=(contract_checks_for(recipes) if contract_checks_for else contract_checks))
    elif step == "exclude_source":
        import re as _re
        m = _re.match(r"\s*['\"`]?([A-Za-z0-9_]+)", note or "")
        name = m.group(1) if m else ""
        present = sorted({k for r in recipes for k, v in r.items() if k != "agg" and v})
        if name not in present:
            print("  [review] exclude_source: %r is not a source in the current space %s; nothing to remove"
                  % (name, present), flush=True)
        else:
            _supersede("recipe_space_%s.json" % safe)
            recipes = _restrict_recipe_space(recipes, name)
            a.exclude_sources = ",".join(t for t in [a.exclude_sources or "", name] if t)
            import hashlib as _hl
            import json as _js
            payload = {"ctx": CTX, "recipes": recipes,
                       "decision": "review excluded source %r: %s" % (name, (note or "")[:600]),
                       "excluded_sources": a.exclude_sources}
            payload["sha256"] = _hl.sha256(_js.dumps(payload, sort_keys=True).encode()).hexdigest()
            os.makedirs(a.freeze_dir, exist_ok=True)
            with open(os.path.join(a.freeze_dir, "recipe_space_%s.json" % safe), "w") as fh:
                _js.dump(payload, fh, indent=2)
            contract, probes = _contract_and_probes(recipes)
            print("  [review] excluded source %r -> %d recipes remain over %s; loader kept"
                  % (name, len(recipes), sorted({k for r in recipes for k, v in r.items() if k != 'agg' and v})),
                  flush=True)
    elif step == "clf_space":
        _supersede("clf_space_%s.json" % safe)
        clf_configs = propose_clf_space(proposer, fb + nprof, CTX, a.freeze_dir, traj, kind=kind)
    elif step == "adapter":
        contract, probes = _contract_and_probes(recipes)
        fp = _adapter_freeze_path(a.freeze_dir, safe)
        if os.path.exists(fp):
            os.replace(fp, fp + ".superseded_by_review")
        load, code, _d, _r = agent.create_adapter(
            {"train_root": a.data_root, "test_root": a.data_root},
            corpus_spec(CTX, extractors, contract + chr(10) + fb, task=_ykey()) + extra_spec, max_iters=a.adapter_iters,
            freeze_path=fp, probes=probes, contract_checks=(contract_checks_for(recipes) if contract_checks_for else contract_checks))
        if load is None:
            print("  [review] the revised adapter did not work; keeping the previous one", flush=True)
    return recipes, clf_configs, load, code, contract, probes


def _finish(traj, safe, t0):
    import time as _time
    el = _time.time() - t0
    ms = getattr(traj, "info", {}).get("model_stats", {}) if hasattr(traj, "info") else {}
    cost = ms.get("cost")
    _stage("6/6 finished", traj)
    print("  [cost] %s / %s: %.1f min wall clock, %s LLM calls, $%s"
          % (CTX, TASK, el / 60.0, ms.get("api_calls", "?"),
             ("%.3f" % cost) if isinstance(cost, (int, float)) else "?"), flush=True)
    traj.log("run_cost", dataset=CTX, task=TASK, wall_clock_min=round(el / 60.0, 2),
             llm_calls=ms.get("api_calls"), usd=cost)
    out = traj.info.get('config', {}).get('output_dir', '')
    traj.close(os.path.join(out, "trajectory_%s_%s.json" % (safe, TASK)))


def _adapter_freeze_path(freeze_dir, safe):
    existing = os.path.join(freeze_dir, "adapter_edaic.py")
    if safe == "E-DAIC" and os.path.exists(existing):
        return existing
    return os.path.join(freeze_dir, "adapter_%s.py" % safe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True,
                    help="corpus root; its basename names the run unless --ctx is given")
    ap.add_argument("--ctx", default=None,
                    help="corpus name used for freeze files and logs (default: basename of data_root)")
    ap.add_argument("--llm_model", default="claude-opus-4-8")
    ap.add_argument("--proposer", default="anthropic", choices=["anthropic"])
    ap.add_argument("--llm_api_key", default=None)
    ap.add_argument("--llm_base_url", default=None)
    ap.add_argument("--freeze_dir", default=None,
                    help="where proposals are frozen (default: artifacts/<ctx>_space)")
    ap.add_argument("--task", default="binary",
                    help="which label this run predicts: a task name the adapter returns as "
                         "train_<task>. Names matching --regression_tasks are fitted as continuous "
                         "targets; anything else as a classification")
    ap.add_argument('--evaluation', choices=['development', 'final'], default='development',
                    help='development runs the full agent on train/dev only; final reports the held-out evaluation')
    ap.add_argument('--output_dir', default=None, help='fresh directory for this complete Agent run')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--development_folds', type=int, default=3)
    ap.add_argument('--max_evaluations', type=int, default=256,
                    help='total unique recipe/model evaluations across all development rounds')
    ap.add_argument('--review_checks', type=int, default=2)
    ap.add_argument('--research_memory', choices=['off', 'evolving'], default='off',
                    help='evolving distills and revises run-local experience after each review round')
    ap.add_argument('--reserve_schedule', choices=['front', 'equal'], default='front',
                    help='front = the whole revision reserve is unlocked at the first review; equal = '
                         'round r may use up to reserve * r / retrospective, so every round can act')
    ap.add_argument('--review_mode', choices=['roles', 'single_agent', 'random'], default='roles',
                    help="roles = the three-role review with a chair, each participant holding "
                         "different evidence. single_agent = one reviewer holding all of it, with "
                         "the chair's rules, budget and verdict grammar unchanged - the ablation of "
                         "role separation itself. random = no review: each round widens a space "
                         "with a uniform draw (the fixed-exploration control; memory must be off).")
    ap.add_argument('--review_accept_policy', choices=['stop', 'require_revision'], default='stop',
                    help="stop = an ACCEPT from the review ends the revision loop (the deployment "
                         "behaviour). require_revision = the fixed-horizon RSI setting: while "
                         "revision budget remains an ACCEPT is answered with one demand for the most "
                         "promising revision, and a round without one adds nothing but does not end "
                         "the loop.")
    ap.add_argument('--reliability_gate', choices=['on', 'off'], default='on',
                    help="measure whether the development ranking survives a held-out fold and "
                         "fall back to 1-SE selection where it does not; 'off' ablates that "
                         "capability rather than turning off a diagnostic")
    ap.add_argument('--selection_rule', choices=['incumbent', 'one_se_of_best'],
                    default='incumbent',
                    help="incumbent = a challenger must beat the current holder by --promotion_se "
                         "standard errors, so the outcome depends on evaluation order; "
                         "one_se_of_best = the classic 1-SE rule over the whole pool, the simplest "
                         "candidate within one SE of the best score, which does not.")
    ap.add_argument('--revision_reserve', type=float, default=None,
                    help='Evaluation budget held back for the rounds after a review, as a fraction '
                         'of --max_evaluations and ADDED to it. The initial search keeps its full '
                         'budget; a run that holds no review gets the reserve in round one instead, '
                         'so runs with and without review are allowed the same total. '
                         'Unset uses DEFAULT_REVISION_RESERVE in development_agent.py (0.25).')
    ap.add_argument('--search_mode',
                    choices=['joint', 'staged', 'fixed_recipe', 'fixed_model', 'no_search', 'no_cpe'],
                    default='joint',
                    help="joint = every recipe x model combination is scored. staged = one probe "
                         "model ranks the recipes, then models are searched on the winning recipe. "
                         "fixed_recipe = the warm-start recipe is carried over and only models are "
                         "searched. fixed_model = the corpus probe estimator is used for every "
                         "recipe and only recipes are searched. no_search = both axes held at those "
                         "same neutral values, one candidate, the floor of the ablation "
                         "ladder. no_cpe = no exploration at all: the improver drives from the "
                         "incumbent and its additions are scored against the incumbent only. "
                         "The last five are the ablations of joint selection for "
                         "CPE; each records how many evaluations it "
                         "actually spent.")
    ap.add_argument('--promotion_se', type=float, default=0.0,
                    help='incumbent rule: a challenger must beat the incumbent by this many subject-bootstrap '
                         'standard errors of the incumbent score (0 = any margin above 1e-12)')
    ap.add_argument('--require_llm', action='store_true',
                    help='fail if no successful LLM execution or requested review completed')
    ap.add_argument('--selection_metric', default=None,
                    help='fixed benchmark metric; when absent the existing metric proposer is used')
    ap.add_argument('--reference_candidates', default=None,
                    help='explicit recipe/config pairs to re-evaluate first on this development venue')
    ap.add_argument("--retrospective", type=int, default=0, metavar="N",
                    help="after the search, hold a three-role review of the decisions and let it "
                         "revise at most N of them before the test is read (0 = no review)")
    ap.add_argument("--adapter_iters", type=int, default=6,
                    help="attempts the adapter agent gets per derivation; a many-source corpus "
                         "can need more than the default 6")
    ap.add_argument("--log", default="auto",
                    help="mirror all output to this file as well as the screen; auto = "
                         "logs/<ctx>_<task>_<timestamp>.log; off = screen only")
    ap.add_argument("--exclude_sources", default="",
                    help="comma-separated source names to drop from the recipe space before "
                         "searching; use to report a restricted-modality condition (e.g. without "
                         "the questionnaire a diagnosis was assigned from)")
    ap.add_argument("--regression_tasks", default="phq,score,severity,total",
                    help="comma-separated task names to treat as continuous targets")
    a = ap.parse_args()
    if a.evaluation == 'final' and a.retrospective:
        ap.error('retrospective is development-only; repeated outer-CV feedback is not final evaluation')
    if a.research_memory == 'evolving' and (a.evaluation != 'development' or a.retrospective < 1):
        ap.error('evolving research memory requires the full development search with retrospective >= 1')
    if a.retrospective < 0 or a.review_checks < 0 or a.max_evaluations < 1 or a.development_folds < 2:
        ap.error('invalid search/review budget')
    import time as _time
    _t0 = _time.time()
    global TASK, CTX, REG_TASKS
    TASK = a.task
    REG_TASKS = tuple(t.strip() for t in a.regression_tasks.split(",") if t.strip())
    CTX = a.ctx or os.path.basename(os.path.normpath(a.data_root))
    safe = CTX.replace("/", "_")
    if not a.freeze_dir:
        a.freeze_dir = os.path.join("artifacts", "%s_space" % safe.lower())

    if a.evaluation == 'development':
        import shutil
        import uuid
        from pathlib import Path
        seed_dir = Path(a.freeze_dir).resolve()
        output = Path(a.output_dir or os.path.join('results', 'full_agent_' + safe.lower(),
                      'seed%d_%s_%s' % (a.seed, _time.strftime('%Y%m%dT%H%M%S'), uuid.uuid4().hex[:8]))).resolve()
        if output.exists():
            ap.error('--output_dir must be a fresh path; existing experiment outputs are preserved')
        if output == seed_dir or output in seed_dir.parents or seed_dir in output.parents:
            ap.error('output and frozen seed directories must not contain each other')
        output.mkdir(parents=True)
        frozen = output / 'frozen'
        if seed_dir.is_dir():
            shutil.copytree(seed_dir, frozen, ignore=shutil.ignore_patterns('__pycache__'))
        else:
            frozen.mkdir()
        a.freeze_dir, a.output_dir = str(frozen), str(output)
    global _REPORT_DIR
    _REPORT_DIR = a.output_dir
    if _REPORT_DIR:
        os.makedirs(_REPORT_DIR, exist_ok=True)

    if a.log != "off":
        from merid.utils.progress import tee_output as _tee
        _tee(a.log if a.log != "auto" else os.path.join(
            a.output_dir or "logs", "%s_%s_%s.log" % (safe.lower(), TASK, _time.strftime("%Y%m%d_%H%M%S"))))

    proposer = ClaudeProposer(model=a.llm_model, provider=a.proposer, base_url=a.llm_base_url,
                              api_key=a.llm_api_key)
    agent = AdapterAgent(proposer)
    traj = Trajectory(model=a.llm_model, config={"corpus": CTX, "task": TASK,
                      'evaluation': a.evaluation, 'output_dir': a.output_dir or ''},
                      stream_path=os.path.join(a.output_dir or '', "trajectory_%s.jsonl" % safe))
    traj.add_system_prompt("adapter_codegen", ADAPTER_SYSTEM)
    proposer.traj = traj

    extractors = discover_extractors(a.data_root)
    print(f"########## {CTX} ##########", flush=True)
    print(f"  extractors discovered: {extractors}", flush=True)
    if not extractors:
        print("  (no extractor directories — this corpus ships recordings, so the adapter has to "
              "compute features itself)", flush=True)

    _stage("1/6 inspect the corpus", traj)
    report = inspect_dataset({"train_root": a.data_root})
    print("== inspection == (%d chars; the adapter prints it in full below)" % len(report), flush=True)

    global SEL_NAME
    kind = "regression" if _is_reg() else "classification"
    task_desc = ("continuous-severity REGRESSION on the '%s' target" % TASK if _is_reg()
                 else "classification of the '%s' label" % TASK)
    if a.selection_metric:
        from merid.run_method import CLF_METRIC_NAMES, REG_METRIC_NAMES
        if a.selection_metric not in (REG_METRIC_NAMES if _is_reg() else CLF_METRIC_NAMES):
            ap.error('selection_metric is unsupported for this task kind')
        SEL_NAME = a.selection_metric
        traj.log('benchmark_metric_fixed', choice=SEL_NAME, dataset=CTX, task=TASK)
    else:
        _stage("2/6 decide the benchmark metric", traj, detail="may search the web")
        SEL_NAME = propose_eval_metric(proposer, CTX, task_desc, report, a.freeze_dir, traj, kind=kind)

    _stage("3/6 propose the recipe space", traj)
    recipes = propose_recipe_space(proposer, report, extractors, a.freeze_dir, traj)
    proposed_count = len(recipes)
    proposed_recipes = list(recipes)
    try:
        recipes = _restrict_recipe_space(recipes, a.exclude_sources)
    except ValueError as exc:
        raise SystemExit("[%s] %s" % (CTX, exc)) from exc
    if a.exclude_sources:
        drop = {t.strip() for t in a.exclude_sources.split(",") if t.strip()}
        print("  [restricted] excluding source(s) %s -> %d recipes (from %d)"
              % (sorted(drop), len(recipes), proposed_count), flush=True)
    import json as _json
    _adapter_vetoes, _recipe_widths = [], {}
    contract, probes = _loader_contract(recipes, proposed_recipes, a.exclude_sources)

    _stage("4/6 derive the adapter", traj, detail="up to 6 iterations; each decodes the corpus")
    load, code, data, adapter_reasoning = agent.create_adapter(
        {"train_root": a.data_root, "test_root": a.data_root},
        corpus_spec(CTX, extractors, contract, task=_ykey()), max_iters=a.adapter_iters,
        freeze_path=_adapter_freeze_path(a.freeze_dir, safe), probes=probes,
        data_scope='development' if a.evaluation == 'development' else 'all',
        adapter_cache_bytes=256 * 1024 * 1024 if a.evaluation == 'development' else 0,
        require_row_ids=a.evaluation == 'development')
    traj.log("adapter_codegen", reasoning=adapter_reasoning, dataset=CTX, ok=load is not None, code=code)
    if load is None and adapter_reasoning:
        print("  [recipe-space] the adapter could not work with the proposed space; re-proposing once "
              "with what it reported", flush=True)
        _nl = chr(10)
        fb = ("A PREVIOUS ATTEMPT FAILED and you must take account of it. The loader written against "
              "your last recipe space reported:" + _nl + str(adapter_reasoning)[-1200:] + _nl + _nl +
              "Propose recipes using the usable sources the data actually supports. Sources "
              "declared excluded by the user must remain excluded, including equivalent renamed "
              "column groups: %s. Do not add a source reported as unusable. "
              "Where usable data is a single table, separate allowed column groups by what they "
              "measure and vary them independently when possible. Include combinations of allowed "
              "sources only; do not restore excluded questionnaire or target-defining columns to "
              "satisfy source counts or table coverage. A single usable source is acceptable; do "
              "not invent additional sources." % (a.exclude_sources or "none") + _nl + _nl)
        fp_space = os.path.join(a.freeze_dir, "recipe_space_%s.json" % CTX.replace("/", "_"))
        if os.path.exists(fp_space):
            os.replace(fp_space, fp_space + ".superseded")
        recipes = propose_recipe_space(proposer, report, extractors, a.freeze_dir, traj, feedback=fb)
        proposed_recipes = list(recipes)
        try:
            recipes = _restrict_recipe_space(recipes, a.exclude_sources)
        except ValueError as exc:
            raise SystemExit("[%s] %s" % (CTX, exc)) from exc
        contract, probes = _loader_contract(recipes, proposed_recipes, a.exclude_sources)
        load, code, data, adapter_reasoning = agent.create_adapter(
            {"train_root": a.data_root, "test_root": a.data_root},
            corpus_spec(CTX, extractors, contract, task=_ykey()), max_iters=a.adapter_iters,
            freeze_path=_adapter_freeze_path(a.freeze_dir, safe), probes=probes,
            data_scope='development' if a.evaluation == 'development' else 'all',
            adapter_cache_bytes=256 * 1024 * 1024 if a.evaluation == 'development' else 0,
            require_row_ids=a.evaluation == 'development')

    if load is None:
        raise SystemExit(
            "adapter FAILED on %s. That is a reportable result, not a crash to paper over: the spec "
            "names no layout, no delimiter and no label location, so failing here means the agent "
            "could not derive them from the inspection on a corpus it has not seen." % CTX)

    if a.evaluation == 'development':
        from merid.core.development_agent import run_development
        _stage("5/6 development search + review", traj)
        run_development(sys.modules[__name__], a, proposer, load, recipes, report,
                        extractors, adapter_reasoning, traj, adapter_source=code)
        _finish(traj, safe, _t0)
        print('RESULT_DIR=' + a.output_dir, flush=True)
        return

    probe, skipped = None, []
    for rc in recipes:
        try:
            probe = load(rc)
            break
        except NotImplementedError as exc:
            skipped.append((rc, "VETOED: " + str(exc)[:150]))
            _adapter_vetoes.append(str(exc)[:220])
        except Exception as exc:
            skipped.append((rc, "%s: %s" % (type(exc).__name__, str(exc)[:120])))
    for rc, why in skipped:
        print("  [recipe %s] %s" % (rc, why), flush=True)
    if probe is None:
        raise SystemExit(
            "[%s] not one of the %d proposed recipes could be loaded. The lines above give each "
            "reason; where they are vetoes, the recipe space named sources the adapter found it could "
            "not honour, and a re-run will propose a fresh space from the same inspection."
            % (CTX, len(recipes)))
    have = sorted(k[len("train_"):] for k in probe
                  if k.startswith("train_") and k not in ("train_X", "train_subject"))
    if _ykey() not in have:
        raise SystemExit(
            "[%s] this corpus defines no '%s' label. The adapter returned: %s. "
            "Re-run with --task <one of those>. The adapter is entitled to REFUSE a task the corpus "
            "does not define rather than synthesise one, and where it has done so the refusal IS the "
            "finding: asking again for the same missing label will not conjure it."
            % (CTX, TASK, ", ".join(have) or "no labels at all"))
    HAS_TEST = ("test_X" in probe and ("test_%s" % _ykey()) in probe
                and np.asarray(probe["test_X"]).shape[0] != np.asarray(probe["train_X"]).shape[0])
    if not HAS_TEST:
        if "test_X" in probe:
            print("  [protocol] the adapter returned a test block with the same row count as train — "
                  "treating this corpus as having NO published split rather than measuring the "
                  "training data twice", flush=True)
        import collections as _c
        _X = np.asarray(probe["train_X"], float)
        _yv = _y(probe, "train")
        _g = np.asarray(probe.get("train_subject", np.arange(len(_yv))))
        nprof = ("%s: n=%d examples from %d subjects, p=%d; %s; selection is a subject-disjoint "
                 "nested CV (this corpus publishes no held-out split)"
                 % (CTX, _X.shape[0], len(np.unique(_g)), _X.shape[1],
                    ("target mean=%.2f sd=%.2f" % (_yv.mean(), _yv.std())) if _is_reg()
                    else ("label dist %s" % dict(sorted(_c.Counter(np.asarray(_yv, int).tolist()).items())))))
        print("  REGIME: %s" % nprof, flush=True)
        clf_configs = propose_clf_space(proposer, nprof, CTX, a.freeze_dir, traj, kind=kind)
        cg_build, _cg = ((None, None) if _is_reg() else
                         propose_codegen_method(proposer, nprof, CTX, a.freeze_dir, traj))
        res = None
        _prior_rounds = []
        for _rnd in range(1 + max(0, a.retrospective)):
            _stage("5/6 nested cross-validation", traj)
            res = nested_report(load, recipes, clf_configs, cg_build, traj,
                                widths=_recipe_widths)
            if res is None:
                raise SystemExit(
                    "[%s] nested cross-validation produced no result. The reason is the line printed "
                    "directly above this one — read that rather than guessing from this message." % CTX)
            if _rnd >= a.retrospective:
                break
            from merid.core.evidence_guided_evolution.retrospective import (
                hold_review, write_minutes)
            _yt, _yp, _n, _picks = res
            import collections as _c2
            ev = _selection_evidence(
                recipes, adapter_reasoning, _adapter_vetoes, _recipe_widths, _picks,
                "subject-disjoint nested CV, %d folds" % len(_picks), a.freeze_dir, CTX,
                extra={"out_of_fold_prediction_distribution":
                           dict(sorted(_c2.Counter(np.asarray(_yp).tolist()).items())),
                       "out_of_fold_truth_distribution":
                           dict(sorted(_c2.Counter(np.asarray(_yt).tolist()).items())),
                       "note": "these out-of-fold predictions ARE the selection venue; no test data "
                               "has been read"})
            verdict, minutes = hold_review(
                proposer, ev, tools=_review_tools(load, recipes, report),
                prior_rounds=_prior_rounds or None, traj=traj, ctx=CTX)
            write_minutes(a.freeze_dir, CTX, TASK, _rnd + 1, minutes)
            _prior_rounds.append({"round": _rnd + 1, "minutes": minutes[-2500:],
                                  "decision": (verdict or {"step": "ACCEPT"})})
            if verdict is None:
                break
            recipes, clf_configs, _newload, _newcode, contract, probes = _apply_revision(
                verdict, proposer, report, extractors, contract, probes, agent, a, traj,
                recipes, clf_configs, kind, nprof, safe, load_now=load)
            if _newload is not None:
                load = _newload
        yt, yp, nrep, picks = res
        _report(yt, yp, nrep, "nested subject-disjoint CV", picks, traj)
        _finish(traj, safe, _t0)
        return

    best = select_recipe(load, recipes, traj)
    if best is None:
        raise SystemExit(
            f"[{CTX}] none of the {len(recipes)} proposed recipes produced usable features. The "
            f"per-recipe lines above give the reason for each; if they all report constant or "
            f"empty features, the adapter is reading the CSVs wrongly (delimiter, header, or "
            f"metadata columns) rather than the recipes being bad.")
    best_score, best_recipe, data = best
    VENUE = "dev" if _has_dev(data) else "grouped-OOF"
    print(f"  [RECIPE PICKED] {best_recipe} {VENUE}-{SEL_NAME}={best_score:.4f}", flush=True)

    X = np.asarray(data["train_X"], float)
    Xt = np.asarray(data["test_X"], float)
    y = _y(data, "train")
    Xd = np.asarray(data["dev_X"], float) if _has_dev(data) else None
    yd = _y(data, "dev") if _has_dev(data) else None
    import collections
    ndev = (Xd.shape[0] if Xd is not None else 0)
    prof = (f"{CTX}: n_train={X.shape[0]} / n_dev={ndev} / n_test={Xt.shape[0]} examples, "
            f"p={X.shape[1]}; " + (f"train target mean={y.mean():.2f} sd={y.std():.2f} "
                                   f"range[{y.min():.0f},{y.max():.0f}]" if _is_reg() else
                                   f"train dist {dict(sorted(collections.Counter(y.tolist()).items()))}"))
    traj.log("regime_profile", dataset=CTX, profile=prof)
    print(f"  REGIME: {prof}", flush=True)

    mod_priors = propose_modality_priors(proposer, report, prof, best_recipe, CTX, a.freeze_dir, traj)
    mc = modality_credibility(load, best_recipe, mod_priors, traj)
    if mc:
        print(f"  [modality-signal] full {VENUE}-{SEL_NAME}={mc['full']:.4f}", flush=True)
        for src, v in mc["modalities"].items():
            pw = v.get("prior_weight")
            print(f"      {src:<10} loo={v['loo']} marginal={v['marginal']:+.4f} -> {v['verdict']}"
                  + (f"   (agent prior {pw}, {v.get('prior_conf')})" if pw is not None else ""),
                  flush=True)
        if mc.get("n_tested"):
            print(f"      prior-vs-measured: {mc['n_agree']}/{mc['n_tested']} agree", flush=True)
        traj.log("modality_signal", dataset=CTX, task=TASK, observation=mc)

    clf_configs = propose_clf_space(proposer, prof, CTX, a.freeze_dir, traj, kind=kind)
    if _is_reg():
        print("  [codegen] skipped on the regression track (the codegen prompt and its smoke test "
              "are classifier-shaped) — this run searches the proposed regressor space only",
              flush=True)
        codegen_build, codegen_code = None, None
    else:
        codegen_build, codegen_code = propose_codegen_method(proposer, prof, CTX, a.freeze_dir, traj)
    cfg, dev_score, yp = select_model(data, Xt, clf_configs, codegen_build, traj)
    print(f"  [CLF PICKED] {cfg} {VENUE}-{SEL_NAME}={dev_score:.4f}", flush=True)

    ydp = _fit_predict(cfg, codegen_build, X, y, Xd) if Xd is not None else None
    yt = _y(data, "test")
    BASELINES = {
        ("E-DAIC", "binary"): "  Flex-MoE (A+V+T, same test): 0.7143 | 0.5958 | 0.5993 | 0.6836 | 0.2209",
        ("E-DAIC", "phq"): ("  baselines (same test): MulT 6.9902|5.6641|-0.0047|-0.2022   "
                            "Flex-MoE 7.0394|5.6417|+0.0135|-0.2192   MDAgents 6.6201|5.3482|+0.0520|-0.0783"),
    }
    ref = BASELINES.get((CTX, TASK))
    sel_line = (f"  DEV  (n={len(yd)})" if ydp is not None
                else f"  SELECT (grouped-OOF in train, n={len(y)})")
    print("", flush=True)
    print(f"==================== MERID - {CTX} / {TASK} ====================", flush=True)
    if _is_reg():
        from merid.run_method import metric_value as _mv

        def row(yy, pp):
            return (-_mv("neg_rmse", yy, pp), -_mv("neg_mae", yy, pp),
                    _mv("ccc", yy, pp), _mv("r2", yy, pp))
        rmse, mae, ccc, r2 = row(yt, yp)
        print(f"  columns: RMSE | MAE | CCC | R2   (selection optimised {SEL_NAME} on {VENUE})",
              flush=True)
        if ydp is not None:
            d0, d1, d2, d3 = row(yd, ydp)
            print(f"{sel_line}  {d0:.4f} | {d1:.4f} | {d2:+.4f} | {d3:+.4f}", flush=True)
        else:
            print(f"{sel_line}  {SEL_NAME}={dev_score:+.4f}", flush=True)
        print(f"  TEST (n={len(yt)})  {rmse:.4f} | {mae:.4f} | {ccc:+.4f} | {r2:+.4f}", flush=True)
        print(f"  selection->test gap on {SEL_NAME}: {_mv(SEL_NAME, yt, yp) - dev_score:+.4f}",
              flush=True)
        if ref:
            print(ref, flush=True)
        const = np.full(len(yt), float(y.mean()))
        print(f"  predict-the-training-mean: {-_mv('neg_rmse', yt, const):.4f} | "
              f"{-_mv('neg_mae', yt, const):.4f} | +0.0000 | {_mv('r2', yt, const):+.4f}", flush=True)
        traj.log("result", dataset=CTX, task=TASK, venue=VENUE,
                 observation={"rmse": round(rmse, 4), "mae": round(mae, 4), "ccc": round(ccc, 4),
                              "r2": round(r2, 4), "selection_score": round(float(dev_score), 4)})
    else:
        from sklearn.metrics import cohen_kappa_score
        acc, wacc, mf1, wf1, off = official(yt, yp)
        kap = float(cohen_kappa_score(yt, yp))
        print(f"  columns: Acc | Bal-Acc | Macro-F1 | Weighted-F1 | Kappa   "
              f"(selection optimised {SEL_NAME} on {VENUE})", flush=True)
        if ydp is not None:
            da, dw, dm, dwf, _ = official(yd, ydp)
            dk = float(cohen_kappa_score(yd, ydp))
            print(f"{sel_line}  {da:.4f} | {dw:.4f} | {dm:.4f} | {dwf:.4f} | {dk:.4f}", flush=True)
        else:
            print(f"{sel_line}  {SEL_NAME}={dev_score:.4f}", flush=True)
        print(f"  TEST (n={len(yt)})  {acc:.4f} | {wacc:.4f} | {mf1:.4f} | {wf1:.4f} | {kap:.4f}",
              flush=True)
        print(f"  selection->test gap on {SEL_NAME}: "
              f"{metric_value(SEL_NAME, yt, yp) - dev_score:+.4f}", flush=True)
        if ref:
            print(ref, flush=True)
        maj = int(np.bincount(np.asarray(y, int)).argmax())
        cy = np.full(len(yt), maj, int)
        ca, cw, cm, cwf, _ = official(yt, cy)
        print(f"  predict-the-training-majority: {ca:.4f} | {cw:.4f} | {cm:.4f} | {cwf:.4f} | "
              f"{float(cohen_kappa_score(yt, cy)):.4f}", flush=True)
        traj.log("result", dataset=CTX, task=TASK, venue=VENUE,
                 observation={"acc": round(acc, 4), "bal_acc": round(wacc, 4),
                              "macro_f1": round(mf1, 4), "weighted_f1": round(wf1, 4),
                              "kappa": round(kap, 4), "composite": round(off, 4),
                              "selection_score": round(float(dev_score), 4)})
    _finish(traj, safe, _t0)


if __name__ == "__main__":
    main()
