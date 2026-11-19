"""Pin the v0.34 packed-scale allocation fix and its CPU build gate."""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "38206ccd55f93a0079bad588e67bc96c08bb9ed6f91e86566811f0ac3ae169f3"


class PackedStoragePatchTests(unittest.TestCase):
    def test_patch_and_cpu_gate_are_in_build(self):
        patch = ROOT / "patches/humming-packed-bs-storage-v034.patch"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), DIGEST)
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(DIGEST, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)
        self.assertIn("python3 /tmp/check_humming_packed_storage.py", dockerfile)

    def test_only_moe_conversion_is_changed(self):
        text = (ROOT / "patches/humming-packed-bs-storage-v034.patch").read_text()
        paths = [line for line in text.splitlines() if line.startswith("diff --git")]
        self.assertEqual(len(paths), 1)
        self.assertIn("quantization/utils/humming/moe.py", paths[0])
        self.assertNotIn("model_executor/utils.py", paths[0])
        self.assertIn("+    if not config.use_packed_bs:", text)
        self.assertIn("+    if direct_reload_enabled():", text)
        self.assertIn("+    param.data = packed.detach()", text)
        self.assertIn("+        packed = packed.clone()", text)

    def test_cpu_gate_parses_and_checks_allocation_not_only_shape(self):
        source = (ROOT / "tools/check_humming_packed_storage.py").read_text()
        ast.parse(source)
        self.assertIn("from vllm.model_executor.utils import get_live_param, replace_parameter", source)
        self.assertIn("original.untyped_storage().nbytes(), 32", source)
        self.assertIn("original.untyped_storage().nbytes(), 64", source)
        for test in ("parameter_aliases_loader", "disabled_keeps_stock", "view_is_copied",
                     "repeat_does_not_change", "direct_reload_refused",
                     "noncontiguous_scale_rejected", "retired_parameter_rejected"):
            self.assertIn(test, source)


if __name__ == "__main__":
    unittest.main()
