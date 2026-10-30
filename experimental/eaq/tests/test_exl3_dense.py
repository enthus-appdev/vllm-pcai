"""Stream handling for EXL3 dense reconstruction; CPU only, reconstruction is faked."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcai_eaq.exl3_dense import dense_modules, dequantize_stream


def spec(module, exl3=True):
    if not exl3:
        return {"stored_tensors": {module + ".weight": {}}}
    return {"quant_format": "exl3",
            "stored_tensors": {f"{module}.{p}": {} for p in ("trellis", "suh", "svh", "mul1")}}


STORAGE = {
    "head": spec("head"),
    "layers.0.attn.wq_b": spec("layers.0.attn.wq_b"),
    "layers.0.attn.wq_a": spec("layers.0.attn.wq_a", exl3=False),
    **{f"layers.0.attn.wo_a.slice.{g}": spec(f"layers.0.attn.wo_a.slice.{g}") for g in range(3)},
    "layers.0.ffn.experts.7.w1": spec("layers.0.ffn.experts.7.w1"),
    "mtp.0.attn.wq_b": spec("mtp.0.attn.wq_b"),
}


def fake(parts):
    # Encode the module identity in the value so concatenation order is checkable.
    return parts["trellis"].float().expand(2, 4).clone()


def parts(module, value):
    return [(f"{module}.{p}", torch.tensor([[value]], dtype=torch.int16)) for p in ("mul1", "svh", "trellis", "suh")]


class DenseStream(unittest.TestCase):
    def test_selects_dense_exl3_only(self):
        modules = dense_modules(STORAGE)
        self.assertEqual(sorted(modules), ["head", "layers.0.attn.wo_a.slice.0", "layers.0.attn.wo_a.slice.1",
                                           "layers.0.attn.wo_a.slice.2", "layers.0.attn.wq_b"])

    def test_stream(self):
        modules = dense_modules(STORAGE)
        # Slices arrive out of order and interleaved with other tensors.
        stream = (parts("layers.0.attn.wo_a.slice.2", 2) + [("embed.weight", torch.zeros(1))]
                  + parts("layers.0.attn.wo_a.slice.0", 0) + parts("head", 9)
                  + parts("layers.0.ffn.experts.7.w1", 7) + parts("layers.0.attn.wo_a.slice.1", 1)
                  + [("layers.0.attn.wq_a.weight", torch.ones(1))] + parts("mtp.0.attn.wq_b", 5))
        out = list(dequantize_stream(iter(stream), modules, reconstruct=fake))
        names = [n for n, _ in out]
        self.assertIn("head.weight", names)
        self.assertIn("layers.0.attn.wo_a.weight", names)
        self.assertNotIn("head.trellis", names)
        self.assertFalse(any(".slice." in n for n in names))
        # Routed experts, draft layers and BF16 tensors pass through untouched.
        self.assertEqual(sum(n.startswith("layers.0.ffn.experts.7.w1.") for n in names), 4)
        self.assertEqual(sum(n.startswith("mtp.0.attn.wq_b.") for n in names), 4)
        self.assertIn("layers.0.attn.wq_a.weight", names)
        wo_a = dict(out)["layers.0.attn.wo_a.weight"]
        self.assertEqual(tuple(wo_a.shape), (6, 4))
        self.assertEqual(wo_a[:, 0].tolist(), [0, 0, 1, 1, 2, 2])

    def test_incomplete_module_fails(self):
        modules = dense_modules(STORAGE)
        stream = parts("head", 1)[:3]
        with self.assertRaisesRegex(ValueError, "incomplete EXL3 modules"):
            list(dequantize_stream(iter(stream), modules, reconstruct=fake))

    def test_missing_slice_fails(self):
        modules = dense_modules(STORAGE)
        stream = parts("layers.0.attn.wo_a.slice.0", 0) + parts("layers.0.attn.wo_a.slice.1", 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            list(dequantize_stream(iter(stream), modules, reconstruct=fake))

    def test_unexpected_parts_rejected(self):
        bad = {"x": {"quant_format": "exl3", "stored_tensors": {"x.trellis": {}, "x.weird": {}}}}
        with self.assertRaises(ValueError):
            dense_modules(bad)


if __name__ == "__main__":
    unittest.main()

