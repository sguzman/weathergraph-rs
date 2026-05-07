from __future__ import annotations

import importlib.util
import json
import pathlib
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "export_weights.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_weights", MODULE_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportWeightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_normalize_flat_key_haiku_segments(self) -> None:
        normalized = self.module.normalize_flat_key(
            "one_step_fn/edge_update_fn_encoder/~/linear_1/w"
        )
        self.assertEqual(normalized, "one_step_fn.edge_update_fn_encoder.linear_1.w")

    def test_alias_key_maps_upstream_encoder_layer_norm(self) -> None:
        mapped = self.module.alias_key("one_step_fn.edge_update_fn_encoder.~.layer_norm.scale")
        self.assertEqual(mapped, "encoder_edge_mlp.layer_norm.weight")

    def test_remap_keys_reports_unmapped_entries(self) -> None:
        tensors = {
            "one_step_fn.edge_update_fn_encoder.~.linear.w": np.eye(2, dtype=np.float32),
            "unknown_module.w": np.ones((1,), dtype=np.float32),
        }
        remapped, unmapped = self.module.remap_keys(tensors, {}, auto_alias=True)

        self.assertIn("encoder_edge_mlp.layers.0.weight", remapped)
        self.assertEqual(unmapped, ["unknown_module.w"])

    def test_remap_keys_transposes_linear_weights_for_rust_contract(self) -> None:
        tensors = {
            "one_step_fn.edge_update_fn_encoder.~.linear.w": np.arange(
                6, dtype=np.float32
            ).reshape(2, 3),
        }
        remapped, unmapped = self.module.remap_keys(tensors, {}, auto_alias=True)

        self.assertEqual(unmapped, [])
        np.testing.assert_array_equal(
            remapped["encoder_edge_mlp.layers.0.weight"],
            np.arange(6, dtype=np.float32).reshape(2, 3).T,
        )

    def test_flatten_params_handles_nested_mappings_and_sequences(self) -> None:
        payload = {
            "params": [
                {"layer": np.eye(2, dtype=np.float32)},
                {"bias": np.zeros(2, dtype=np.float32)},
            ]
        }
        flattened = self.module.flatten_params(payload)

        self.assertIn("params.0.layer", flattened)
        self.assertIn("params.1.bias", flattened)

    def test_dry_run_cli_emits_unmapped_report_without_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            payload = {
                ("one_step_fn/edge_update_fn_encoder/~/linear", "w"): np.eye(
                    2, dtype=np.float32
                ),
                ("unmapped_module", "w"): np.ones((1,), dtype=np.float32),
            }
            source = temp_path / "weights.pkl"
            out = temp_path / "weights.safetensors"
            unmapped = temp_path / "unmapped.json"
            with source.open("wb") as handle:
                pickle.dump(payload, handle)

            result = subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--source",
                    str(source),
                    "--out",
                    str(out),
                    "--emit-unmapped",
                    str(unmapped),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("Dry run mapped 2 tensors", result.stderr)
            self.assertFalse(out.exists())
            unmapped_payload = json.loads(unmapped.read_text(encoding="utf-8"))
            self.assertEqual(unmapped_payload["unmapped_raw_keys"], ["unmapped_module.w"])

    def test_dry_run_cli_emits_mapping_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = pathlib.Path(tmpdir)
            payload = {
                ("unmapped_module", "w"): np.ones((1,), dtype=np.float32),
                ("other_unmapped", "b"): np.zeros((1,), dtype=np.float32),
            }
            source = temp_path / "weights.pkl"
            out = temp_path / "weights.safetensors"
            mapping_template = temp_path / "mapping-template.json"
            with source.open("wb") as handle:
                pickle.dump(payload, handle)

            subprocess.run(
                [
                    sys.executable,
                    str(MODULE_PATH),
                    "--source",
                    str(source),
                    "--out",
                    str(out),
                    "--emit-mapping-template",
                    str(mapping_template),
                    "--dry-run",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            template_payload = json.loads(mapping_template.read_text(encoding="utf-8"))
            self.assertEqual(
                template_payload,
                {
                    "other_unmapped.b": "",
                    "unmapped_module.w": "",
                },
            )


if __name__ == "__main__":
    unittest.main()
