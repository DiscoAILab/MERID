"""Catalog of released methods available to the agent."""
from __future__ import annotations

def _svc(C=0.1, k=128, kernel="linear"):
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return Pipeline([("v", VarianceThreshold(0.0)), ("s", StandardScaler()),
                     ("k", SelectKBest(f_classif, k=k)), ("m", SVC(C=C, kernel=kernel, class_weight="balanced"))])


def _pls_head(k=128, nc=2):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_regression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("v", VarianceThreshold(0.0)), ("s", StandardScaler()),
                     ("k", SelectKBest(f_regression, k=k)), ("m", PLSRegression(n_components=nc))])


REGISTRY: dict = {
    "sklearn_svc": {
        "kind": "classical_head", "source": "sklearn", "released": True, "builder": _svc,
        "modality_input": "one concatenated feature matrix X[n,p]",
        "regime_fit": "SMALL n, p>>n — strong default; robust with SelectKBest+class_weight",
        "params": "C, k, kernel(linear/rbf)"},
    "sklearn_logreg": {
        "kind": "classical_head", "source": "sklearn", "released": True,
        "modality_input": "X[n,p]", "regime_fit": "small n, linear, well-calibrated", "params": "C, k"},
    "sklearn_gb": {
        "kind": "classical_head", "source": "sklearn", "released": True,
        "modality_input": "X[n,p]", "regime_fit": "small n, nonlinear, low depth", "params": "lr, depth, k"},
    "sklearn_rf_et": {
        "kind": "classical_head", "source": "sklearn", "released": True,
        "modality_input": "X[n,p]", "regime_fit": "small n, robust to noise; can overfit p>>n", "params": "n_estimators"},
    "sklearn_pls": {
        "kind": "classical_reduce_head", "source": "sklearn", "released": True, "builder": _pls_head,
        "modality_input": "X[n,p]", "regime_fit": "p>>n workhorse (supervised dim reduction) — PHQ regression + threshold",
        "params": "k, n_components"},

    "competition_framework": {
        "kind": "classical_pipeline", "source": "mpdd2026_data.py + mpdd2026_heads.py", "released": True,
        "modality_input": "per-event [PAIR,T,D] audio/video + gait + personality (MPDD format)",
        "regime_fit": "MPDD-AVG family; 3-head recombine (Young) / stratmeta stacking (Elder)",
        "params": "feature bank, PLS-n, shrink, gate/severity, stack_meta"},

    "flex_moe": {
        "kind": "deep_fusion_missing_modality", "source": "external/Flex-MoE", "released": True,
        "repo_url": "https://github.com/UNITES-Lab/Flex-MoE.git",
        "modality_input": "per-modality feature vectors + a MISSING-modality mask; G-Router needs some "
                          "full-modality samples, S-Router specializes to observed subsets",
        "regime_fit": "MISSING modalities / cross-dataset pooling; needs more data + FastMoE/CUDA",
        "params": "num_experts, hidden, top_k; entry models.py/moe_module.py"},
    "mult": {
        "kind": "deep_fusion_crossmodal", "source": "external/Multimodal-Transformer", "released": True,
        "repo_url": "https://github.com/yaohungt/Multimodal-Transformer.git",
        "modality_input": "THREE modality SEQUENCES x_l,x_a,x_v each [batch, seq_len, feat_dim] "
                          "(l=text/personality, a=audio, v=visual); Conv1d proj -> cross-modal transformer",
        "regime_fit": "temporal multimodal, deep — needs enough data (overfits on n~90)",
        "params": "num_heads, layers, dropouts; entry src/models.py MULTModel + src/train.py"},
}


def catalog_text(kinds=None, names=None) -> str:
    lines = []
    for name, m in REGISTRY.items():
        if names is not None and name not in names:
            continue
        if kinds and m["kind"] not in kinds:
            continue
        lines.append(f"- {name} [{m['kind']}] src={m['source']}\n"
                     f"    input: {m['modality_input']}\n"
                     f"    suits: {m['regime_fit']}")
    return "\n".join(lines)


SELECT_SYSTEM = (
    "You are an AutoML research agent. Given a dataset's MULTIMODAL PROFILE and a catalog of REAL "
    "released methods (source code available), SELECT the single best method for THIS dataset — or "
    "answer CREATE if none in the catalog fits (then a new method will be codegen'd, grounded by "
    "retrieved papers/repos). Reason from the regime: n (small -> classical, not deep), p>>n, which "
    "modalities are present, missingness, temporal sequences vs pooled vectors, cross-dataset pooling. "
    "Output your reasoning, then the chosen method NAME (or the word CREATE) on the FINAL line.")


def select_method(profile_text, proposer, *, available_methods=None):
    if available_methods is not None:
        names = list(available_methods)
        if (not names or len(names) != len(set(names)) or
                any(name not in REGISTRY for name in names)):
            raise ValueError('available_methods must contain unique registered executable methods')
        system = (
            'You are an AutoML research agent. Select one method from the executable catalog '
            'for this dataset profile. The invoking runner has explicitly declared which '
            'implementations it can execute. Reason from the observed regime and return the '
            'chosen method NAME alone on the FINAL line. Choose only an available catalog name.')
        prompt = (f'DATASET PROFILE:\n{profile_text}\n\nEXECUTABLE METHOD SPACE:\n'
                  f'{catalog_text(names=names)}\n\nWhich ONE method best fits THIS dataset? '
                  'Reason, then output its exact name on the final line.')
        text = proposer.complete(system, prompt)
        final = (text or '').strip().splitlines()
        selected = final[-1].strip() if final else ''
        if selected not in names:
            raise ValueError('method selection did not name an available executable method')
        return selected, text
    prompt = (f"DATASET PROFILE:\n{profile_text}\n\nMETHOD SPACE (real released repos):\n{catalog_text()}\n\n"
              "Which ONE method best fits THIS dataset? Reason, then output the name (or CREATE) as the final line.")
    txt = proposer.complete(SELECT_SYSTEM, prompt)
    names = list(REGISTRY) + ["CREATE"]
    for line in reversed((txt or "").splitlines()):
        for n in names:
            if n.lower() in line.lower():
                return n, txt
    return "sklearn_svc", txt


def ensure_repo(name, work_dir="external"):
    import os
    import subprocess
    m = REGISTRY.get(name, {})
    src = m.get("source", "")
    if src.startswith("external/") and os.path.isdir(src):
        return src
    url = m.get("repo_url")
    if not url:
        return src if os.path.isdir(src) else None
    dest = os.path.join(work_dir, os.path.basename(url).replace(".git", ""))
    if os.path.isdir(dest):
        return dest
    os.makedirs(work_dir, exist_ok=True)
    try:
        subprocess.run(["git", "clone", "--depth", "1", url, dest], check=True,
                       capture_output=True, timeout=300)
        print(f"  [clone] {name}: git cloned {url} -> {dest}", flush=True)
        return dest
    except Exception as exc:
        print(f"  [clone] {name}: FAILED to clone {url} ({type(exc).__name__}) — method unavailable", flush=True)
        return None


def get(name):
    return REGISTRY.get(name)


def builders():
    return {n: m["builder"] for n, m in REGISTRY.items() if "builder" in m}
