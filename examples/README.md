# Runnable Example

This example creates synthetic input files, trains for one epoch on the CPU, and evaluates the saved checkpoint. It uses eight samples and 16 x 16 grids to keep the run small. All model branches use the same channel counts and nine time steps as the main workflow.

The data is random and has no physical meaning. The resulting scores do not measure tropical cyclone forecast skill.

## Run from the Project Folder

```bash
python -m pip install -r requirements.txt
python examples/create_sample_data.py
python TCSF-Main.py --config examples/config.smoke.json --check-data
python TCSF-Main.py --config examples/config.smoke.json --mode train
python TCSF-Main.py --config examples/config.smoke.json --mode evaluate
```

The generator creates two Excel files and four pickle caches in `examples/sample_data/`. It refuses to replace existing input files unless you add `--overwrite`.

Training writes the best checkpoint and evaluation results to `examples/outputs/`. Run training before evaluation, or place a compatible checkpoint at the configured checkpoint path first.

Paths inside a configuration file are relative to that file's directory. You can run the scripts from another working directory by passing absolute script and config paths.

## Use Your Own Data

Edit the project-level `config.json` to point to your preprocessed Excel files and caches. Its default grid size is 201 x 201 and its default training duration is 100 epochs.

Do not mix the generated placeholder features with real meteorological data. Real inputs must follow the auxiliary feature order and channel meanings used by the original training data.
