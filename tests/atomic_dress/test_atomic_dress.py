"""Numerical regressions; run: python -m unittest discover -s tests/atomic_dress."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bin"))
from weight_utils import fit_residual_dress, verify_residual_dress


class AtomicDressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        self.config = {
            "model": {"name": "potential_model", "params": {
                "e_dress": {1: -6.0, 8: -3.0, 79: -10.0},
                "e_unit": 1.0, "e_scale": 100.0, "use_e_weight": True}},
            "network": {"name": "PiNet2"},
            "optimizer": {"class_name": "Adam"},
            "model_dir": "original",
        }
        self.write(self.config)

    def write(self, config):
        (self.model / "params.yml").write_text(yaml.safe_dump(config))

    def read(self):
        return yaml.safe_load((self.model / "params.yml").read_text())

    def test_fixed_composition_additive_fit_and_preservation(self):
        # Bias 2 eV/cell plus centered errors. A replacement would lose -15 eV.
        report = fit_residual_dress(self.model, [[1, 1, 8]] * 3,
                                    np.array([-13.3, -13, -12.7]) / 3,
                                    np.full(3, -15.0) / 3)
        dress = self.read()["model"]["params"]["e_dress"]
        self.assertAlmostEqual(2 * dress[1] + dress[8], -13.0)
        self.assertEqual(dress[79], -10.0)
        # Null-space vector (1,-2) is unchanged by a minimum-norm correction.
        self.assertAlmostEqual(dress[1] - 2 * dress[8], 0.0)
        self.assertEqual(report["rank"], 1)
        self.assertAlmostEqual(report["rmse_per_atom_after"], np.sqrt(2 / 3) * 0.1)
        config = self.read()
        config["model"]["params"]["e_dress"] = self.config["model"]["params"]["e_dress"]
        self.assertEqual(config, self.config)
        verify_residual_dress(self.model)

    def test_varying_counts_and_output_units(self):
        config = copy.deepcopy(self.config)
        config["model"]["params"]["e_unit"] = 27.2114
        self.write(config)
        # Known per-element correction: H +1, O +2 (in label units).
        fit_residual_dress(self.model, [[1], [8], [1, 8]],
                           [-5, -1, -3], np.array([-6, -3, -4.5]) * 27.2114)
        dress = self.read()["model"]["params"]["e_dress"]
        self.assertAlmostEqual(dress[1], -5)
        self.assertAlmostEqual(dress[8], -1)
        self.assertEqual(dress[79], -10)

    def test_overwrite_is_detected(self):
        fit_residual_dress(self.model, [[1, 1, 8]], [-4], [-5])
        verify_residual_dress(self.model)
        config = self.read()
        config["model"]["params"]["e_dress"][1] += 0.1
        self.write(config)
        with self.assertRaisesRegex(ValueError, "overwrote"):
            verify_residual_dress(self.model)

    def test_invalid_data_does_not_write(self):
        before = (self.model / "params.yml").read_bytes()
        for elements, labels, predictions in [
            ([], [], []), ([[1]], [float("nan")], [0]),
            ([[1]], [0, 1], [0]), ([[]], [0], [0]),
        ]:
            with self.subTest(elements=elements):
                with self.assertRaises(ValueError):
                    fit_residual_dress(self.model, elements, labels, predictions)
                self.assertEqual((self.model / "params.yml").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
