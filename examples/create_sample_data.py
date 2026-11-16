"""Create a small, deterministic synthetic dataset for testing the TCSF workflow.

The generated values have no physical interpretation and cannot be used to
assess forecast accuracy. Relative data paths are resolved from the config file.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


QUADRANTS = ("NW", "NE", "SW", "SE")
THRESHOLDS = (34, 50, 64)
NUM_TIMESTEPS = 9
FEATURES_PER_TIMESTEP = 37


def resolve_path(config_dir, value):
    """Resolve a configured path without depending on the working directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each data path must be a nonempty string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def generate(config_path, overwrite=False):
    """Write two Excel files and four cache dictionaries for eight samples."""
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)

    config_dir = config_path.parent
    data = config["data"]
    image_size = data["image_size"]
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(type(size) is not int or size < 2 for size in image_size)
    ):
        raise ValueError("data.image_size must contain two integers of at least 2.")
    height, width = image_size

    main_path = resolve_path(config_dir, data["main_excel"])
    auxiliary_path = resolve_path(config_dir, data["auxiliary_excel"])
    cache_paths = {
        name: resolve_path(config_dir, data["image_caches"][name])
        for name in ("main", "u", "v", "q")
    }
    output_paths = [main_path, auxiliary_path, *cache_paths.values()]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("All six configured data paths must be different.")
    existing = [path for path in output_paths if path.exists()]
    if existing and not overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing data files:\n"
            + names
            + "\nUse --overwrite only if you want to replace them with synthetic data."
        )

    split_years = config["split_years"]
    ranges = {}
    for split in ("train", "validation", "test"):
        bounds = split_years[split]
        if (
            not isinstance(bounds, list)
            or len(bounds) != 2
            or any(type(year) is not int for year in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError(f"split_years.{split} must be [first_year, last_year].")
        ranges[split] = bounds
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if max(ranges[left][0], ranges[right][0]) <= min(ranges[left][1], ranges[right][1]):
            raise ValueError("Training, validation, and test year ranges must not overlap.")

    # Four training rows, two validation rows, and two test rows.
    years = [ranges["train"][1]] * 4
    years += [ranges["validation"][0]] * 2
    years += [ranges["test"][0]] * 2
    rng = np.random.default_rng(config["training"]["seed"])
    caches = {name: {} for name in cache_paths}
    rows = []
    for sample_index, year in enumerate(years):
        row = {"year": year}
        for step in range(NUM_TIMESTEPS):
            frame_id = f"synthetic_{sample_index:03d}_t{step:02d}"
            row[f"day{step}"] = frame_id
            key = frame_id + ".png"
            for name, cache in caches.items():
                channels = 2 if name == "main" else 4
                cache[key] = rng.random((channels, height, width), dtype=np.float32)
        for quadrant in QUADRANTS:
            base = float(rng.uniform(10.0, 40.0))
            for threshold, offset in zip(THRESHOLDS, (30.0, 15.0, 0.0)):
                row[f"day16_USA_R{threshold}_{quadrant}"] = base + offset
        rows.append(row)

    # Positional placeholder names deliberately make no claim about physical variables.
    auxiliary_columns = [
        f"t{step}_feature{feature:02d}"
        for step in range(NUM_TIMESTEPS)
        for feature in range(FEATURES_PER_TIMESTEP)
    ]
    auxiliary_values = rng.normal(size=(len(rows), len(auxiliary_columns))).astype(np.float32)

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(main_path, index=False, engine="openpyxl")
    pd.DataFrame(auxiliary_values, columns=auxiliary_columns).to_excel(
        auxiliary_path, index=False, engine="openpyxl"
    )
    for name, path in cache_paths.items():
        with path.open("wb") as handle:
            pickle.dump(caches[name], handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Created synthetic data: 8 samples, 9 time steps, 333 auxiliary values per sample.")
    print(f"Grid size: {height} x {width}; channels: main=2, U/V/Q=4 each.")
    print("Synthetic values are for workflow testing only, not forecast evaluation.")
    for path in output_paths:
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().with_name("config.smoke.json"),
        help="JSON configuration file (default: examples/config.smoke.json).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing configured input files with synthetic data.",
    )
    args = parser.parse_args()
    try:
        generate(args.config, overwrite=args.overwrite)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
