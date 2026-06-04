#!/usr/bin/env python3
"""Train-slice scorer for DeepQuark-style method artifacts.

This evaluator reads artifacts from checkpoints/<method>_<system>/ and scores
one half of the stored evaluation records. It does not import method code or
rerun optimization/inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


SLICE = "train"
SLICE_INDEX = 0

EXPERIMENTS = {
    "tcc_al1_binding": {
        "suffix": "ccqq",
        "threshold_mev": 1862.0 + 2016.0,
        "expected_params": 1857,
        "color_keys": ("color_3bar3", "color_6bar6"),
    },
    "tbb_al1_binding": {
        "suffix": "bbqq",
        "threshold_mev": 5294.0 + 5350.0,
        "expected_params": 1857,
        "color_keys": ("color_3bar3", "color_6bar6"),
    },
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _evaluation_records(results: dict) -> list[tuple[str, str, dict[str, Any]]]:
    records = []
    for checkpoint_key, checkpoint_result in sorted(results.items(), key=lambda item: str(item[0])):
        if not isinstance(checkpoint_result, dict):
            continue
        for repeat_key, repeated_eval in sorted(checkpoint_result.items(), key=lambda item: str(item[0])):
            if isinstance(repeated_eval, dict) and "Emean" in repeated_eval:
                records.append((str(checkpoint_key), str(repeat_key), repeated_eval))
    return records


def _best_eval(results: dict) -> dict:
    records = _evaluation_records(results)
    candidates = [record for idx, (_, _, record) in enumerate(records) if idx % 2 == SLICE_INDEX]
    candidates = candidates or [record for _, _, record in records]
    if not candidates:
        raise ValueError("No evaluation records with Emean were found")
    return min(candidates, key=lambda record: float(record["Emean"]))


def _parse_color(text: str, nkeys: int) -> list[float]:
    values = [float(value) for value in re.findall(r":([0-9.]+)", text)]
    if len(values) < nkeys:
        values.extend([math.nan] * (nkeys - len(values)))
    return values[:nkeys]


def _method_dirs(artifact_root: Path, suffix: str) -> dict[str, Path]:
    methods = {}
    for artifact_dir in sorted(artifact_root.glob(f"*_{suffix}")):
        if not artifact_dir.is_dir():
            continue
        if not (artifact_dir / "results.json").exists():
            continue
        if not (artifact_dir / "parameters.json").exists():
            continue
        method_name = artifact_dir.name[: -(len(suffix) + 1)]
        if method_name:
            methods[method_name] = artifact_dir
    return methods


def _baseline(config: dict) -> dict:
    scores = {
        "energy_mev": config["threshold_mev"],
        "binding_mev": 0.0,
        "mc_error_mev": 0.0,
        "parameter_count": 0.0,
    }
    for key in config["color_keys"]:
        scores[key] = 0.0
    return scores


def _score_artifact(artifact_dir: Path, config: dict) -> dict:
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
    parser.add_argument("--output", default="scoring/scores_train.json")
    parser.add_argument(
        "--experiments",
        default=",".join(EXPERIMENTS),
        help="comma-separated experiment names to include in the train-slice scores",
    )
    args = parser.parse_args()

    artifact_root = Path(args.artifacts)
    requested = [name.strip() for name in args.experiments.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(EXPERIMENTS))
    if unknown:
        raise ValueError(f"Unknown experiment(s): {', '.join(unknown)}")

    scored = {"slice": SLICE, "experiments": {}}
    for exp_name in requested:
        config = EXPERIMENTS[exp_name]
        results = {"threshold_baseline": _baseline(config)}
        for method_name, artifact_dir in _method_dirs(artifact_root, config["suffix"]).items():
            results[method_name] = _score_artifact(artifact_dir, config)
        scored["experiments"][exp_name] = {"results": results}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(scored, handle, indent=2)


if __name__ == "__main__":
    main()
