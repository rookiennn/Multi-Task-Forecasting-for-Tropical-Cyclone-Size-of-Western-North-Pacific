"""Integration checks for configuration, sample data, and the CLI workflow.

Run with: python -m unittest discover -s tests -v
Only standard-library modules are imported by the test runner. The application
and its third-party dependencies run in subprocesses using the same interpreter.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "TCSF-Main.py"
GENERATOR_SCRIPT = PROJECT_ROOT / "examples" / "create_sample_data.py"


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="tcsf-workflow-", dir=Path(__file__).resolve().parent
        )
        self.addCleanup(self.temporary.cleanup)
        self.case_dir = Path(self.temporary.name)
        self.config_dir = self.case_dir / "configuration"
        self.config_dir.mkdir()
        self.other_cwd = self.case_dir / "different-working-directory"
        self.other_cwd.mkdir()
        self.config = json.loads(
            (PROJECT_ROOT / "examples" / "config.smoke.json").read_text(encoding="utf-8")
        )
        # All generated files stay in the temporary folder. Relative paths and a
        # different subprocess working directory test config-relative resolution.
        self.config["data"]["main_excel"] = "data/samples.xlsx"
        self.config["data"]["auxiliary_excel"] = "data/auxiliary.xlsx"
        self.config["data"]["image_caches"] = {
            "main": "data/img_dict.pkl",
            "u": "data/img_dict_U.pkl",
            "v": "data/img_dict_V.pkl",
            "q": "data/img_dict_Q.pkl",
        }
        self.config["output"]["directory"] = "results"
        self.config["output"]["checkpoint"] = "custom_checkpoint.pth"
        self.config_path = self.config_dir / "config.json"
        self.config_path.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        self.output_dir = self.config_dir / "results"

    def run_python(self, *args, expect_success=True):
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        result = subprocess.run(
            [sys.executable, *map(str, args)],
            cwd=self.other_cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        details = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        if expect_success:
            self.assertEqual(result.returncode, 0, details)
        else:
            self.assertNotEqual(result.returncode, 0, details)
        return result

    def generate(self):
        self.run_python(GENERATOR_SCRIPT, "--config", self.config_path)

    def data_path(self, key):
        return self.config_dir / self.config["data"][key]

    def prediction_snapshot(self):
        result = self.run_python(
            "-c",
            "import pandas as pd, sys; "
            "print(pd.read_excel(sys.argv[1]).to_json(orient='split', double_precision=15))",
            self.output_dir / "predictions.xlsx",
        )
        return json.loads(result.stdout)

    def test_complete_training_and_evaluation_workflow(self):
        self.generate()
        self.run_python(MAIN_SCRIPT, "--config", self.config_path, "--check-data")
        self.assertFalse(self.output_dir.exists(), "Data validation must not create run outputs.")

        self.run_python(MAIN_SCRIPT, "--config", self.config_path, "--mode", "train")
        checkpoint = self.output_dir / self.config["output"]["checkpoint"]
        self.assertTrue(checkpoint.is_file())
        self.assertTrue((self.output_dir / "mae_results.txt").is_file())
        metrics = json.loads((self.output_dir / "metrics.json").read_text(encoding="utf-8"))
        for key in ("mae", "rmse", "r2"):
            self.assertIsInstance(metrics[key], float)
            self.assertTrue(math.isfinite(metrics[key]), f"Metric {key} must be finite.")
        trained_predictions = self.prediction_snapshot()
        self.assertEqual(len(trained_predictions["data"]), 2)
        self.assertEqual(len(trained_predictions["columns"]), 24)
        for row in trained_predictions["data"]:
            self.assertTrue(all(isinstance(value, (int, float)) and math.isfinite(value) for value in row))

        self.run_python(MAIN_SCRIPT, "--config", self.config_path, "--mode", "evaluate")
        self.assertEqual(
            self.prediction_snapshot(),
            trained_predictions,
            "Evaluation must reuse the same saved best checkpoint as the final training evaluation.",
        )

    def test_rejects_wrong_auxiliary_width(self):
        self.generate()
        self.run_python(
            "-c",
            "import pandas as pd, sys; "
            "frame = pd.read_excel(sys.argv[1]); "
            "frame.iloc[:, :-1].to_excel(sys.argv[1], index=False)",
            self.data_path("auxiliary_excel"),
        )
        result = self.run_python(
            MAIN_SCRIPT, "--config", self.config_path, "--check-data", expect_success=False
        )
        self.assertIn("333", result.stdout + result.stderr)
        self.assertFalse(self.output_dir.exists())

    def test_rejects_missing_cache_key(self):
        self.generate()
        cache_path = self.config_dir / self.config["data"]["image_caches"]["main"]
        self.run_python(
            "-c",
            "import pickle, pathlib, sys; "
            "path = pathlib.Path(sys.argv[1]); "
            "cache = pickle.loads(path.read_bytes()); "
            "cache.pop(next(iter(cache))); "
            "path.write_bytes(pickle.dumps(cache))",
            cache_path,
        )
        result = self.run_python(
            MAIN_SCRIPT, "--config", self.config_path, "--check-data", expect_success=False
        )
        self.assertIn("cache", (result.stdout + result.stderr).lower())
        self.assertFalse(self.output_dir.exists())

    def test_generator_refuses_to_overwrite_inputs(self):
        self.generate()
        paths = [self.data_path("main_excel"), self.data_path("auxiliary_excel")]
        paths += [self.config_dir / value for value in self.config["data"]["image_caches"].values()]
        original_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        result = self.run_python(
            GENERATOR_SCRIPT, "--config", self.config_path, expect_success=False
        )
        self.assertIn("overwrite", (result.stdout + result.stderr).lower())
        for path in paths:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), original_hashes[path])

    def test_import_has_no_runtime_side_effects(self):
        result = self.run_python(
            "-c",
            "import importlib.util, pathlib, sys; "
            "path = pathlib.Path(sys.argv[1]); "
            "sys.path.insert(0, str(path.parent)); "
            "spec = importlib.util.spec_from_file_location('tcsf_import_check', path); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module)",
            MAIN_SCRIPT,
        )
        self.assertEqual(result.stdout.strip(), "")
        self.assertEqual(list(self.other_cwd.iterdir()), [])
        self.assertFalse(self.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
