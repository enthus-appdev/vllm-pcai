"""CPU checks of packed backing storage; not GPU/model qualification.

Execute the helper AST from installed patched vLLM, using real PyTorch tensors
and vLLM's real parameter replacement. Avoid importing GPU-only MoE backends.
"""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from vllm.model_executor.utils import get_live_param, replace_parameter


def load_helper():
    root = Path(importlib.util.find_spec("vllm").origin).parent
    # Optional source overlay for a CPU Job testing a patch against the old image.
    path = Path(os.environ.get("PCAI_TEST_MOE_SOURCE", str(
        root / "model_executor/layers/quantization/utils/humming/moe.py")))
    source = path.read_text()
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_compact_packed_scale_storage")
    future = ast.parse("from __future__ import annotations").body[0]
    module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
    scope = {"torch": torch, "get_live_param": get_live_param,
             "logger": SimpleNamespace(info_once=lambda *a: None)}
    exec(compile(module, "installed-moe-storage-helper", "exec"), scope)
    prepare = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_prepare_and_transform_sublayer")
    calls = [n.value.func.id for n in prepare.body
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name)]
    assert calls[-2:] == ["_replace_layer_parameters", "_compact_packed_scale_storage"], calls
    return scope[function.name]


compact = load_helper()


def fixture(packed=None):
    layer = torch.nn.Module()
    original = torch.nn.Parameter(torch.arange(64, dtype=torch.uint8).reshape(2, 32),
                                  requires_grad=False)
    original.weight_loader = object()
    layer.register_parameter("w13_weight_scale", original)
    layer.register_parameter("checkpoint_scale_alias", original)
    if packed is None:
        packed = torch.full((2, 16), 7, dtype=torch.uint8)
    replace_parameter(layer, "w13_weight_scale", packed)
    return layer, original, packed


class PackedStorageTests(unittest.TestCase):
    def apply(self, layer, tensor, enabled=True):
        compact(layer, "w13", SimpleNamespace(use_packed_bs=enabled), {"weight_scale": tensor})

    def test_unpatched_replacement_retains_original_capacity(self):
        layer, original, tensor = fixture()
        self.assertEqual(original.numel(), 32)
        self.assertEqual(original.untyped_storage().nbytes(), 64)
        self.assertTrue(torch.equal(original, tensor))

    def test_compact_storage_preserves_parameter_aliases_loader_and_values(self):
        layer, original, tensor = fixture()
        loader = original.weight_loader
        self.apply(layer, tensor)
        self.assertEqual(original.untyped_storage().nbytes(), 32)
        self.assertIs(layer.w13_weight_scale, original)
        self.assertIs(layer.checkpoint_scale_alias, original)
        self.assertIs(original.weight_loader, loader)
        self.assertTrue(torch.equal(original, tensor))
        self.assertFalse(original.requires_grad)

    def test_disabled_keeps_stock_storage(self):
        layer, original, tensor = fixture()
        ptr = original.data_ptr()
        self.apply(layer, tensor, enabled=False)
        self.assertEqual(original.data_ptr(), ptr)
        self.assertEqual(original.untyped_storage().nbytes(), 64)

    def test_view_is_copied_into_exact_storage(self):
        for offset in (0, 8):
            tensor = torch.full((128,), 9, dtype=torch.uint8)[offset:offset + 32].reshape(2, 16)
            layer, original, tensor = fixture(tensor)
            self.apply(layer, tensor)
            self.assertEqual(original.storage_offset(), 0)
            self.assertEqual(original.untyped_storage().nbytes(), 32)
            self.assertTrue(torch.equal(original, tensor))

    def test_repeat_does_not_change_compact_pointer(self):
        layer, original, tensor = fixture()
        self.apply(layer, tensor)
        ptr = original.data_ptr()
        self.apply(layer, tensor)
        self.assertEqual(original.data_ptr(), ptr)

    def test_direct_reload_refused(self):
        layer, original, tensor = fixture()
        with patch("vllm.model_executor.model_loader.reload.direct.direct_reload_enabled",
                   return_value=True):
            with self.assertRaisesRegex(ValueError, "direct weight reload"):
                self.apply(layer, tensor)
        self.assertEqual(original.untyped_storage().nbytes(), 64)

    def test_noncontiguous_scale_rejected(self):
        layer, original, tensor = fixture()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            self.apply(layer, tensor.t())

    def test_retired_parameter_rejected(self):
        layer, original, tensor = fixture()
        original.retired = True
        with self.assertRaisesRegex(TypeError, "live"):
            self.apply(layer, tensor)


if __name__ == "__main__":
    unittest.main(verbosity=2)
