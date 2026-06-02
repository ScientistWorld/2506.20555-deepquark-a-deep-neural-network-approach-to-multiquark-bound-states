#!/usr/bin/env python3
"""Score DeepQuark-style artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


EXPERIMENTS = {
    "tcc_al1_binding": {
        "artifact": "deepquark_ccqq",
        "method": "deepquark",
        "threshold_mev": 1862.0 + 2016.0,
        "expected_params": 1889,
        "color_keys": ("color_3bar3", "color_6bar6"),
    },
    "tbb_al1_binding": {
        "artifact": "deepquark_bbqq",
        "method": "deepquark",
        "threshold_mev": 5294.0 + 5350.0,
        "expected_params": 1889,
        "color_keys": ("color_3bar3", "color_6bar6"),
    },
    "pcc_al1_binding": {
        "artifact": "deepquark_ccqqc",
        "method": "deepquark",
        "threshold_mev": 2016.0 + 3702.0,
        "expected_params": 2921,
        "color_keys": ("color_3bar3", "color_3bar6", "color_6bar3"),
    },
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _best_eval(results: dict) -> dict:
    best = None
    for checkpoint_result in results.values():
        if not isinstance(checkpoint_result, dict):
            continue
        for repeated_eval in checkpoint_result.values():
            if not isinstance(repeated_eval, dict) or "Emean" not in repeated_eval:
                continue
            energy = float(repeated_eval["Emean"])
            if best is None or energy < float(best["Emean"]):
                best = repeated_eval
    if best is None:
        raise ValueError("No evaluation records with Emean were found")
    return best


def _parse_color(text: str, nkeys: int) -> list[float]:
    values = []
    for value in re.findall(r":([0-9.]+)", text):
        values.append(float(value))
    if len(values) < nkeys:
        values.extend([math.nan] * (nkeys - len(values)))
    return values[:nkeys]


def score_artifact(artifact_dir: Path, exp_name: str) -> dict:
    config = EXPERIMENTS[exp_name]
    result = _best_eval(_load_json(artifact_dir / "results.json"))
    params = _load_json(artifact_dir / "parameters.json")

    energy_mev = float(result["Emean"]) * 1000.0
    sigma_mev = float(result.get("Esigma", 0.0)) * 1000.0
    binding_mev = energy_mev - config["threshold_mev"]
    para_num = float(params.get("hyperparams", {}).get("para_num", config["expected_params"]))
    colors = _parse_color(result.get("color", ""), len(config["color_keys"]))

    scores = {
        "energy_mev": energy_mev,
        "binding_mev": binding_mev,
        "mc_error_mev": sigma_mev,
        "parameter_count": para_num,
    }
    for key, value in zip(config["color_keys"], colors):
        scores[key] = value
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="checkpoints", help="directory containing method artifacts")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    artifact_root = Path(args.artifacts)
    scored = {"experiments": {}}
    for exp_name, config in EXPERIMENTS.items():
        baseline = {
            "energy_mev": config["threshold_mev"],
            "binding_mev": 0.0,
            "mc_error_mev": 0.0,
            "parameter_count": 0.0,
        }
        for key in config["color_keys"]:
            baseline[key] = 0.0
        scored["experiments"][exp_name] = {"results": {"threshold_baseline": baseline}}

        artifact_dir = artifact_root / config["artifact"]
        if not (artifact_dir / "results.json").exists():
            continue
        scored["experiments"][exp_name]["results"][config["method"]] = score_artifact(artifact_dir, exp_name)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(scored, handle, indent=2)


if __name__ == "__main__":
    main()
