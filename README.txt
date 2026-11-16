# TCSF: Tropical Cyclone Size Forecasting

This PyTorch project predicts tropical cyclone wind radii 24 hours ahead using meteorological grids and numerical features from nine historical time steps.

Each forecast contains 12 values: R34, R50, and R64 for four quadrants, ordered NW, NE, SW, SE. Within each quadrant, the order is R34, R50, R64.

## 1. Project Files

| File | Purpose |
| --- | --- |
| `TCSF-Main.py` | Loads data, trains the model, evaluates a checkpoint, and saves results. |
| `ObFeature.py` | Extracts features from historical quadrant values and environmental variables. |
| `TC3DFeature.py` | Extracts temporal and spatial features from meteorological grids using 3D convolutions. |
| `requirements.txt` | Pinned Python package versions for the reference environment. |
| `config.json` | Paths and training settings for your own data. |
| `examples/` | A small configuration, synthetic data generator, and runnable example. |
| `model/` | The supplied model checkpoint and MAE results. |
| `upload_github.ps1` | Commits the project and pushes it to an existing GitHub repository. |

## 2. Environment and Dependencies

Use 64-bit Python 3.12. Install the dependencies in your Python environment:

```bash
python -m pip install -r requirements.txt
```

The reference packages are PyTorch 2.9.1, NumPy 2.3.5, Pandas 3.0.1, scikit-learn 1.7.2, Pillow 12.3.0, and OpenPyXL 3.1.5. These versions define the reference environment; the original research environment was not supplied.

CPU execution is supported. For a CUDA build, follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for the matching PyTorch version. Set `training.device` to `auto`, `cpu`, or `cuda`; `auto` uses CUDA when available.

## 3. Run the Small Example

Run these commands from the project folder:

```bash
python examples/create_sample_data.py
python TCSF-Main.py --config examples/config.smoke.json --check-data
python TCSF-Main.py --config examples/config.smoke.json --mode train
python TCSF-Main.py --config examples/config.smoke.json --mode evaluate
```

The generator creates eight synthetic samples in `examples/sample_data/`. The example trains for one epoch on the CPU using 16 x 16 grids. It exercises the complete data, training, checkpoint, and evaluation workflow. The random data has no physical meaning, and its scores do not measure forecast skill.

Results are saved to `examples/outputs/`. The generator refuses to overwrite existing inputs unless you explicitly add `--overwrite`. See `examples/README.md` for more details.

## 4. Configure Your Own Run

Edit `config.json`. All relative paths are resolved from the configuration file's directory, so there is no need to edit paths inside Python files.

| Setting | Purpose |
| --- | --- |
| `data.main_excel` | Sample table with image identifiers, years, and target values. |
| `data.auxiliary_excel` | Auxiliary numerical features, in matching sample order. |
| `data.image_caches` | Paths to the `main`, `u`, `v`, and `q` pickle caches. |
| `data.image_size` | Frame height and width. The real-data default is `[201, 201]`; frames must already have this size. |
| `training` | Epochs, batch size, learning rate, weight decay, seed, device, worker count, and CPU thread count. |
| `split_years` | Inclusive first and last year for each data split. |
| `output.directory` | Where to save checkpoints and evaluation results. |
| `output.checkpoint` | Checkpoint filename inside the output directory. |

Defaults are 100 epochs, batch size 16, and learning rate 0.0001. Year ranges are 2003-2019 for training, 2020-2021 for validation, and 2022-2023 for testing.

Validate data, train, or evaluate an existing checkpoint:

```bash
python TCSF-Main.py --config config.json --check-data
python TCSF-Main.py --config config.json --mode train
python TCSF-Main.py --config config.json --mode evaluate
```

`--check-data` validates the configured inputs and exits without training. Use `--mode evaluate --check-data` to check a test-only dataset. For command-line help, run `python TCSF-Main.py --help`.

Training saves the checkpoint with the lowest validation RMSE, then reloads that same file for the final test. Evaluation mode loads the configured checkpoint without training. To evaluate the supplied checkpoint, place it at the configured output checkpoint path first. Output directories are created automatically.

## 5. Input Data Format

Real meteorological input data and its preprocessing scripts are not included. Prepare two Excel files and four pickle dictionaries:

| Input | Required contents |
| --- | --- |
| Main Excel file | `year`, `day0` through `day8`, and the 12 columns `day16_USA_R{threshold}_{quadrant}`, using thresholds 34, 50, 64 and quadrants NW, NE, SW, SE. |
| Auxiliary Excel file | Exactly 333 numerical columns: 37 features for each of nine time steps. Rows must match the main table in order. Do not export a DataFrame index column. |
| Main cache | A dictionary of frames with shape `(2, height, width)`. |
| U, V, Q caches | Three dictionaries, each with frames of shape `(4, height, width)`. |

For each `day0`-`day8` cell, the cache key is its value converted to a string with `.png` appended. All referenced frames must exist in all four caches. Values must be finite numerical data.

At each time step, the auxiliary features contain 24 quadrant-related values and 13 environmental values. Keep the exact positional order expected by `CycloneFeatureExtractor.forward()` in `ObFeature.py`. The original physical variable names and channel definitions are not fully documented; the synthetic example uses positional placeholder names.

The `day` names are time-step identifiers. For a 24-hour forecast, adjacent indices must be three hours apart, with `day8` as the forecast starting time and `day16` as the target time.

## 6. Saved Results

Each run uses its configured output directory:

- `best_regression_model.pth`: best validation checkpoint, unless another filename is configured.
- `predictions.xlsx`: true and predicted values for all 12 targets on the test set.
- `mae_results.txt`: MAE for each target.
- `metrics.json`: overall test MAE, RMSE, and R-squared.

The loss function is SmoothL1Loss and the optimizer is Adam. Checkpoint selection uses validation RMSE; the test set is evaluated after selection.

To run the workflow checks:

```bash
python -m unittest discover -s tests -v
```

## 7. Upload to GitHub

Install Git and create an empty repository on GitHub. Leave the new repository's README, license, and .gitignore options unchecked so its history is empty. Make sure your Git credentials have permission to push to it.

If Git author details have not been configured, set your own name and email once:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

In PowerShell, replace `YOUR_USERNAME` with your GitHub username. First preview the steps:

```powershell
.\upload_github.ps1 -RepositoryUrl "https://github.com/YOUR_USERNAME/TCSF.git" -DryRun
```

Then upload:

```powershell
.\upload_github.ps1 -RepositoryUrl "https://github.com/YOUR_USERNAME/TCSF.git"
```

You can also use an SSH address such as `git@github.com:YOUR_USERNAME/TCSF.git`. For later updates, run the same script with a new commit message:

```powershell
.\upload_github.ps1 -RepositoryUrl "https://github.com/YOUR_USERNAME/TCSF.git" -CommitMessage "Update project"
```

The script initializes Git when needed, stages files according to `.gitignore`, commits changes, and pushes the current commit to the `main` branch on the remote. Use `-Branch` to choose a different remote branch. It uses the script's folder as the project root; `-ProjectPath` can select another project directory.

The existing model weights are included. Virtual environments, input data, generated example data, and run outputs are excluded by `.gitignore`. The script keeps an existing origin address, refuses a mismatched origin, and does not force-push. If the remote already has its own commits and Git rejects the push, reconcile the histories before retrying.
