"""MPDD-2026 action spaces of the PHQ-regression and hierarchical ternary heads."""
from __future__ import annotations

SPACE = {
    "audio":  ("data_recipe", ["opensmile", "wav2vec", "mfcc"]),
    "video":  ("data_recipe", ["openface", "densenet", "resnet"]),
    "mode":   ("data_recipe", ["stats_q_pz", "stats_q_pl2", "event_stats_q_pz",
                               "stats_pz", "gp_stats_pz"]),
    "model":  ("representation", ["plsn2", "plsn3", "plsn4", "ridgea30.0"]),
    "k":      ("hyperparam", [64, 128, 256]),
    "shrink": ("regularization", [None, 1.2, 1.3, 1.35, 1.4]),
    "binary_rule": ("decision_rule", ["oof_tuned", "label_cut"]),
}

DEFAULT = {"audio": "opensmile", "video": "openface", "mode": "stats_q_pz",
           "model": "plsn4", "k": 256, "shrink": 1.35, "binary_rule": "oof_tuned"}

TERNARY_SPACE = {
    "audio":      ("data_recipe", ["opensmile", "wav2vec", "mfcc"]),
    "video":      ("data_recipe", ["openface", "densenet", "resnet"]),
    "gate_model": ("classifier",  ["svcC0.1", "logregC0.1", "svcC1.0", "lda"]),
    "sev_model":  ("classifier",  ["gb0.05d1", "gb0.03d1", "svcC0.1", "logregC0.1"]),
    "sev_fusion": ("fusion",      [None, "wav2vec:densenet", "opensmile:openface"]),
    "k":          ("hyperparam",  [32, 64, 128]),
    "align_mode": ("data_recipe", ["alignment", "alignment_ext"]),
}
TERNARY_DEFAULT = {"audio": "opensmile", "video": "openface", "gate_model": "svcC0.1",
                   "sev_model": "gb0.05d1", "sev_fusion": None, "k": 64, "align_mode": "alignment"}


def roles() -> dict:
    return {ax: role for ax, (role, _) in SPACE.items()}


RECIPE_ROLES = ("data_recipe", "fusion")
MODEL_ROLES = ("representation", "classifier", "hyperparam", "regularization", "decision_rule")
SEARCH_AXES = ("all", "fixed_recipe", "fixed_model")

_FAMILY_SPACES = (("binary_phq", SPACE, DEFAULT),
                  ("hierarchical_ternary", TERNARY_SPACE, TERNARY_DEFAULT))


def pinned_axes(search_axes):
    if search_axes in (None, "all"):
        return {}
    if search_axes == "fixed_recipe":
        held = RECIPE_ROLES
    elif search_axes == "fixed_model":
        held = MODEL_ROLES
    else:
        raise ValueError("search_axes must be one of %s" % (SEARCH_AXES,))
    return {family: {key: default[key] for key, (role, _choices) in space.items() if role in held}
            for family, space, default in _FAMILY_SPACES}


def free_axes(search_axes):
    pinned = pinned_axes(search_axes)
    return {family: {key: list(choices) for key, (_role, choices) in space.items()
                     if key not in pinned.get(family, {})}
            for family, space, _default in _FAMILY_SPACES}


def choices(axis: str) -> list:
    return list(SPACE[axis][1])


def validate(config: dict) -> dict:
    out = {}
    for ax, (_role, vals) in SPACE.items():
        v = config.get(ax, DEFAULT[ax])
        out[ax] = v if v in vals else DEFAULT[ax]
    return out


def key(config: dict) -> tuple:
    c = validate(config)
    return tuple(c[ax] for ax in SPACE)


def grid(axes: dict | None = None) -> list:
    import itertools
    axes = axes or {ax: vals for ax, (_r, vals) in SPACE.items()}
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]
