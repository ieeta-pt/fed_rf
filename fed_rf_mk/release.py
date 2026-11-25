"""What leaves each site for one level of the disclosure ladder.

``release_level`` takes the values

    0     source site: its model
    1     source site: its model and n_k, n_k^+; every other site: n_k, n_k^+
    2     source site: its model
    3     every site: its model
    3+1   every site: its model, n_k and n_k^+
    3+2   every site: its model
    all   every site: the full result of the single round (model, counts,
          out of bag scores, analysis outputs), for experiments

For 3 and 3+2, a site also sends n_k when the site weights are ``size`` and
n_k^+ when they are ``positives``, since the weights need them. Out of bag
scores leave a site only for Random Forest with out of bag sampling, at the
levels that combine models (3, 3+1, 3+2).

Pure Python, no PySyft, so that the rules can be tested anywhere.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

RELEASE_LEVELS = ("0", "1", "2", "3", "3+1", "3+2", "all")
SOURCE_LEVELS = ("0", "1", "2")

# Keys of the ladder dict that the deployment site receives, per level
_COMMON = ("model_type", "sites", "sources", "models")
_FED = ("weights", "weights_kind", "sampling", "seed", "T_k", "t", "selections", "oob_fallback")
DEPLOYMENT_KEYS = {
    "0": _COMMON,
    "1": _COMMON + ("pi", "pi_k"),
    "2": _COMMON,
    "3": _COMMON + _FED,
    "3+1": _COMMON + _FED + ("pi", "pi_k"),
    "3+2": _COMMON + _FED,
}


def normalize_level(level) -> str:
    s = str(level).strip()
    if s not in RELEASE_LEVELS:
        raise ValueError(f"release_level must be one of {RELEASE_LEVELS}, got {level!r}")
    return s


def site_plan(level, sites: Sequence[str], source_site: Optional[str] = None,
              site_weights: str = "equal", tree_sampling: str = "uniform",
              model_type: str = "rf") -> Dict[str, dict]:
    """For every training site that gets a code request: the function it runs
    (``ml_experiment`` or ``site_counts``) and what it releases (``release``,
    None for the full result of ``all``). Sites left out get no request."""
    level = normalize_level(level)
    sites = list(sites)
    if level == "all":
        return {s: {"function": "ml_experiment", "release": None} for s in sites}
    if level in SOURCE_LEVELS:
        if source_site not in sites:
            raise ValueError(f"release_level {level} needs source_site among {sites}, "
                             f"got {source_site!r}")
        if level == "1":
            plan = {s: {"function": "site_counts", "release": {"n": True, "n_pos": True}}
                    for s in sites if s != source_site}
            plan[source_site] = {"function": "ml_experiment",
                                 "release": {"model": True, "n": True, "n_pos": True,
                                             "oob": False}}
            return {s: plan[s] for s in sites}
        return {source_site: {"function": "ml_experiment",
                              "release": {"model": True, "n": False, "n_pos": False,
                                          "oob": False}}}
    oob = model_type == "rf" and tree_sampling == "oob"
    if level == "3+1":
        rel = {"model": True, "n": True, "n_pos": True, "oob": oob}
    else:  # 3, 3+2
        rel = {"model": True, "n": site_weights == "size",
               "n_pos": site_weights == "positives", "oob": oob}
    return {s: {"function": "ml_experiment", "release": dict(rel)} for s in sites}


def released_keys(entry: dict) -> set:
    """Keys of the result that a site sends under ``entry`` of ``site_plan``
    (for ``all``: None, the full result)."""
    rel = entry["release"]
    if rel is None:
        return None
    if entry["function"] == "site_counts":
        return {"n_train", "n_pos"}
    keys = {"model_type"}
    for flag, key in (("model", "model"), ("n", "n_train"), ("n_pos", "n_pos"),
                      ("oob", "oob_scores")):
        if rel.get(flag):
            keys.add(key)
    return keys


def deployment_ladder(level, ladder: dict) -> dict:
    """The part of the coordinator's ladder dict that goes to the deployment
    site for this level (everything for ``all``)."""
    level = normalize_level(level)
    if level == "all":
        return dict(ladder)
    return {k: ladder[k] for k in DEPLOYMENT_KEYS[level] if k in ladder}
