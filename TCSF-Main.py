"""Train or evaluate the TCSF model using a JSON configuration file."""

import argparse
import json
from pathlib import Path
import pickle
import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import r2_score

from ObFeature import CycloneFeatureExtractor
from TC3DFeature import TCConv3DFeatureExtractor


NUM_TIMESTEPS = 9
IMAGE_COLUMNS = [f"day{i}" for i in range(NUM_TIMESTEPS)]
TARGET_COLUMNS = [
    f"day16_USA_R{radius}_{quadrant}"
    for quadrant in ("NW", "NE", "SW", "SE")
    for radius in (34, 50, 64)
]
CACHE_CHANNELS = {"main": 2, "u": 4, "v": 4, "q": 4}


def load_image(path, target_size):
    img = Image.open(path).convert("L").resize(target_size)
    return np.array(img, dtype=np.float32) / 255.0


class CycloneDataset(Dataset):
    def __init__(self, df, addition_df, caches):
        self.df = df.reset_index(drop=True)
        self.addition = addition_df.to_numpy(dtype=np.float32, copy=True)
        self.targets = self.df[TARGET_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        self.caches = caches

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        keys = [str(self.df.iloc[idx][column]) + ".png" for column in IMAGE_COLUMNS]
        grids = [
            torch.from_numpy(np.stack([self.caches[name][key] for key in keys]))
            for name in CACHE_CHANNELS
        ]
        return (*grids, torch.from_numpy(self.addition[idx]), torch.from_numpy(self.targets[idx]))


class TCIR_Conv2D(nn.Module):
    def __init__(self, in_channels=2, out_features=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Identity()

    def forward(self, x):
        return torch.flatten(self.features(x), 1)


class TCIR_Module(nn.Module):
    def __init__(self, num_timesteps=9, num_channels=2, num_classes=3):
        super().__init__()
        # Keep these registered layers for compatibility with existing checkpoints.
        # The active image path below uses the 3D convolution branches.
        self.resnet = TCIR_Conv2D(in_channels=num_channels, out_features=512)
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512, nhead=8, dim_feedforward=1024,
                dropout=0.1, batch_first=True,
            ),
            num_layers=2,
        )
        self.conv3d_x = TCConv3DFeatureExtractor(
            in_channels=2, base_channels=16, num_blocks=4, use_se=True,
        )
        self.conv3d_small = TCConv3DFeatureExtractor(
            in_channels=2, base_channels=16, num_blocks=4, use_se=True,
        )
        self.conv3d_u = TCConv3DFeatureExtractor()
        self.conv3d_v = TCConv3DFeatureExtractor()
        self.conv3d_w = TCConv3DFeatureExtractor()
        total_input_dim = 128 + 128 + 64 + 64 + 64 + 128
        self.regressor_q1 = self._make_regressor(total_input_dim, num_classes)
        self.regressor_q2 = self._make_regressor(total_input_dim, num_classes)
        self.regressor_q3 = self._make_regressor(total_input_dim, num_classes)
        self.regressor_q4 = self._make_regressor(total_input_dim, num_classes)
        self.obFeature = CycloneFeatureExtractor(env_dim=13, d_model=128)

    def _make_regressor(self, in_dim, out_dim):
        return nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x, x_u, x_v, x_w, addition):
        batch, timesteps, channels, height, width = x.shape
        quadrants = (
            x[:, :, :, :height // 2, :width // 2],
            x[:, :, :, :height // 2, width // 2:],
            x[:, :, :, height // 2:, :width // 2],
            x[:, :, :, height // 2:, width // 2:],
        )
        quad_features = []
        for patch in quadrants:
            patch_height, patch_width = patch.shape[-2:]
            patch_4d = patch.reshape(batch * timesteps, channels, patch_height, patch_width)
            resized = F.interpolate(
                patch_4d, size=(height, width), mode="bilinear", align_corners=False,
            ).reshape(batch, timesteps, channels, height, width)
            quad_features.append(self.conv3d_small(resized.permute(0, 2, 1, 3, 4)))

        global_feat = self.conv3d_x(x.permute(0, 2, 1, 3, 4))
        feature_u = self.conv3d_u(x_u.permute(0, 2, 1, 3, 4))
        feature_v = self.conv3d_v(x_v.permute(0, 2, 1, 3, 4))
        feature_w = self.conv3d_w(x_w.permute(0, 2, 1, 3, 4))
        heads = (self.regressor_q1, self.regressor_q2, self.regressor_q3, self.regressor_q4)
        outputs = []
        for local_feat, head in zip(quad_features, heads):
            fused = torch.cat([global_feat, local_feat, feature_u, feature_v, feature_w], dim=1)
            combined = torch.cat([fused, self.obFeature(addition)], dim=1)
            outputs.append(head(combined))
        return torch.cat(outputs, dim=1)


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _number(value, name, minimum=0, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if value < minimum or (strictly_positive and value == 0):
        comparison = ">" if strictly_positive else ">="
        raise ValueError(f"{name} must be {comparison} {minimum}.")
    return value


def _path(value, name, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(config_path):
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a JSON object.")
    for section in ("data", "training", "split_years", "output"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Configuration section '{section}' must be an object.")

    data = config["data"]
    for key in ("main_excel", "auxiliary_excel"):
        data[key] = _path(data.get(key), f"data.{key}", config_path.parent)
    caches = data.get("image_caches")
    if not isinstance(caches, dict):
        raise ValueError("data.image_caches must define main, u, v, and q paths.")
    for key in CACHE_CHANNELS:
        caches[key] = _path(caches.get(key), f"data.image_caches.{key}", config_path.parent)
    size = data.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("data.image_size must be [height, width].")
    for dimension in size:
        _positive_int(dimension, "data.image_size dimensions", minimum=2)

    training = config["training"]
    defaults = {
        "epochs": 100, "batch_size": 16, "learning_rate": 0.0001,
        "weight_decay": 0.00001, "seed": 42, "device": "auto",
        "num_workers": 0, "cpu_threads": 1,
    }
    for key, default in defaults.items():
        training.setdefault(key, default)
    for key in ("epochs", "batch_size", "cpu_threads"):
        _positive_int(training[key], f"training.{key}")
    for key in ("seed", "num_workers"):
        _positive_int(training[key], f"training.{key}", minimum=0)
    if training["seed"] >= 2**32:
        raise ValueError("training.seed must be less than 2**32.")
    _number(training["learning_rate"], "training.learning_rate", strictly_positive=True)
    _number(training["weight_decay"], "training.weight_decay")
    device = training["device"]
    if not isinstance(device, str):
        raise ValueError("training.device must be 'auto', 'cpu', or a CUDA device such as 'cuda:0'.")
    if device != "auto":
        try:
            parsed_device = torch.device(device)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Invalid training.device: {device}") from error
        if parsed_device.type not in ("cpu", "cuda"):
            raise ValueError("training.device must use CPU or CUDA.")

    ranges = config["split_years"]
    names = ("train", "validation", "test")
    for name in names:
        bounds = ranges.get(name)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"split_years.{name} must contain inclusive [first_year, last_year].")
        for bound in bounds:
            _positive_int(bound, f"split_years.{name}")
        if bounds[0] > bounds[1]:
            raise ValueError(f"split_years.{name}: first year must not exceed last year.")
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            a, b = ranges[first], ranges[second]
            if max(a[0], b[0]) <= min(a[1], b[1]):
                raise ValueError(f"Year ranges for {first} and {second} overlap.")

    output = config["output"]
    output["directory"] = _path(output.get("directory"), "output.directory", config_path.parent)
    checkpoint = output.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("output.checkpoint must be a checkpoint filename.")
    if Path(checkpoint).name != checkpoint or checkpoint in (".", ".."):
        raise ValueError("output.checkpoint must be a filename, not a directory path.")
    return config


def _numeric_frame(frame, name):
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise").astype(np.float32)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must contain only numerical values.") from error
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"{name} contains missing, infinite, or out-of-range values.")
    return numeric


def prepare_data(config, mode):
    data = config["data"]
    main = pd.read_excel(data["main_excel"])
    auxiliary = pd.read_excel(data["auxiliary_excel"])
    required = ["year", *IMAGE_COLUMNS, *TARGET_COLUMNS]
    missing = [column for column in required if column not in main.columns]
    if missing:
        raise ValueError("Main Excel file is missing columns: " + ", ".join(missing))
    if len(main) != len(auxiliary):
        raise ValueError("Main and auxiliary Excel files must have the same number of sample rows.")
    if auxiliary.shape[1] != NUM_TIMESTEPS * 37:
        raise ValueError(f"Auxiliary Excel file must have exactly 333 columns; found {auxiliary.shape[1]}.")
    auxiliary = _numeric_frame(auxiliary, "Auxiliary Excel file")
    main[TARGET_COLUMNS] = _numeric_frame(main[TARGET_COLUMNS], "Target columns")
    years = _numeric_frame(main[["year"]], "Year column")["year"]
    if not np.equal(years, np.floor(years)).all():
        raise ValueError("Year values must be whole numbers.")
    main["year"] = years.astype(np.int64)
    for column in IMAGE_COLUMNS:
        if main[column].isna().any():
            raise ValueError(f"Image identifier column '{column}' contains missing values.")
        main[column] = main[column].map(str)

    splits = {
        name: np.flatnonzero(main["year"].between(*config["split_years"][name])).tolist()
        for name in ("train", "validation", "test")
    }
    needed = ("train", "validation", "test") if mode == "train" else ("test",)
    for name in needed:
        if not splits[name]:
            raise ValueError(f"The {name} split has no samples. Check split_years and the main Excel file.")

    keys = {value + ".png" for column in IMAGE_COLUMNS for value in main[column]}
    caches = {}
    height, width = data["image_size"]
    for name, channels in CACHE_CHANNELS.items():
        path = data["image_caches"][name]
        with path.open("rb") as handle:
            cache = pickle.load(handle)
        if not isinstance(cache, dict):
            raise ValueError(f"Cache '{name}' must contain a dictionary: {path}")
        missing_keys = keys.difference(cache)
        if missing_keys:
            example = sorted(missing_keys)[0]
            raise ValueError(f"Cache '{name}' is missing {len(missing_keys)} frame(s), including '{example}'.")
        expected_shape = (channels, height, width)
        for key in keys:
            try:
                frame = np.asarray(cache[key], dtype=np.float32)
            except (ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"Cache '{name}', frame '{key}' must be a numerical array.") from error
            if frame.shape != expected_shape:
                raise ValueError(
                    f"Cache '{name}', frame '{key}' has shape {frame.shape}; expected {expected_shape}."
                )
            if not np.isfinite(frame).all():
                raise ValueError(f"Cache '{name}', frame '{key}' contains non-finite values.")
            cache[key] = frame
        caches[name] = cache
    return CycloneDataset(main, auxiliary, caches), splits


def evaluate_regression(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            *inputs, target = [value.to(device) for value in batch]
            predicted = model(*inputs)
            predictions.append(predicted.cpu().numpy())
            targets.append(target.cpu().numpy())
    predictions = np.vstack(predictions).astype(np.float64)
    targets = np.vstack(targets).astype(np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("The model produced non-finite predictions. Check the data scale and checkpoint.")
    metrics = {
        "mae": float(np.abs(predictions - targets).mean()),
        "rmse": float(np.sqrt(np.square(predictions - targets).mean())),
        "r2": float(r2_score(targets, predictions, multioutput="uniform_average")) if len(targets) > 1 else None,
    }
    return metrics, predictions, targets


def train_regression(model, train_loader, val_loader, device, training, checkpoint_path):
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"],
    )
    best_rmse = float("inf")
    for epoch in range(training["epochs"]):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            *inputs, target = [value.to(device) for value in batch]
            optimizer.zero_grad()
            loss = criterion(model(*inputs), target)
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is non-finite. Check feature scales and learning_rate.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * target.size(0)
        validation, _, _ = evaluate_regression(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{training['epochs']} | "
            f"Train loss: {total_loss / len(train_loader.dataset):.4f} | "
            f"Validation MAE: {validation['mae']:.4f} | "
            f"Validation RMSE: {validation['rmse']:.4f}",
            flush=True,
        )
        if validation["rmse"] < best_rmse:
            best_rmse = validation["rmse"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}", flush=True)


def save_results(output_dir, metrics, predictions, targets):
    columns = {}
    per_target_mae = {}
    for index, column in enumerate(TARGET_COLUMNS):
        name = column.removeprefix("day16_USA_R")
        columns[f"True_{name}"] = targets[:, index]
        columns[f"Pred_{name}"] = predictions[:, index]
        per_target_mae[f"MAE_{name}"] = float(np.abs(predictions[:, index] - targets[:, index]).mean())
    pd.DataFrame(columns).to_excel(output_dir / "predictions.xlsx", index=False)
    with (output_dir / "mae_results.txt").open("w", encoding="utf-8") as handle:
        handle.write("Mean Absolute Errors (MAE) for each target:\n" + "=" * 40 + "\n")
        for name, value in per_target_mae.items():
            handle.write(f"{name}: {value:.6f}\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({**metrics, "mae_by_target": per_target_mae}, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run(config, mode, check_data=False):
    dataset, splits = prepare_data(config, mode)
    print("Samples: " + ", ".join(f"{name}={len(indices)}" for name, indices in splits.items()), flush=True)
    if check_data:
        print("Data validation passed. No model was created or trained.")
        return

    training = config["training"]
    random.seed(training["seed"])
    np.random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    torch.set_num_threads(training["cpu_threads"])
    device_name = training["device"]
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("A CUDA device was requested, but CUDA is unavailable. Use training.device='cpu' or 'auto'.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is unavailable.")
    output_dir = config["output"]["directory"]
    checkpoint_path = output_dir / config["output"]["checkpoint"]
    if mode == "evaluate" and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Train first or update output settings.")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = TCIR_Module().to(device)
    print(f"Using device: {device}", flush=True)

    def loader(name, shuffle=False):
        return DataLoader(
            Subset(dataset, splits[name]), batch_size=training["batch_size"], shuffle=shuffle,
            num_workers=training["num_workers"],
        )

    if mode == "train":
        train_regression(
            model, loader("train", shuffle=True), loader("validation"), device, training, checkpoint_path,
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    metrics, predictions, targets = evaluate_regression(model, loader("test"), device)
    save_results(output_dir, metrics, predictions, targets)
    print(f"Test metrics: {json.dumps(metrics, allow_nan=False)}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json", help="JSON config path (default: config.json).")
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train", help="Train and test, or evaluate an existing checkpoint.")
    parser.add_argument("--check-data", action="store_true", help="Validate inputs for the selected mode and exit without creating a model.")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        run(config, args.mode, check_data=args.check_data)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()

