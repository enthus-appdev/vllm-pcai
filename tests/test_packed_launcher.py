"""Pin strict launcher shape validation and native-rebuild integration."""
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/humming-packed-bs-launcher.patch"
DIGEST = "4c0ca83172f59a0953f2ad2d2b699a486167679dd384d8c99f65d9324d0e1ae0"


class PackedLauncherTests(unittest.TestCase):
    def test_patch_and_rebuild_are_pinned(self):
        self.assertEqual(hashlib.sha256(PATCH.read_bytes()).hexdigest(), DIGEST)
        docker = (ROOT / "Dockerfile").read_text()
        self.assertIn(DIGEST, docker)
        self.assertIn("MAX_JOBS=1 OMP_NUM_THREADS=1 python3 /tmp/build_humming_launcher.py", docker)

    def test_metadata_is_explicit_and_legacy_cubins_default_false(self):
        text = PATCH.read_text()
        self.assertIn('+  bool use_packed_bs;', text)
        self.assertIn('reader.getBoolOrDefault("USE_PACKED_BS", false)', text)
        self.assertIn('symbolOffsets.count(name) ? getBool(name) : fallback', text)

    def test_half_width_does_not_disable_common_validation(self):
        text = PATCH.read_text()
        self.assertIn('kernel_data.problem_shape_n / (kernel_data.use_packed_bs ? 2 : 1)', text)
        self.assertIn('!kernel_data.use_block_scaled_mma && !kernel_data.use_tma_bs', text)
        self.assertIn('kernel_data.problem_shape_n % 32 == 0', text)
        self.assertNotIn('-  check_tensor_common', text)

    def test_nibble_addressing_contract(self):
        for n in (128, 256, 2048, 4096):
            self.assertEqual((n // 2) % 16, 0)
            for low in range(16):
                for high in range(16):
                    packed = low | (high << 4)
                    expanded_word = (packed & 15) | ((packed & 240) << 4)
                    self.assertEqual(expanded_word, low | (high << 8))


if __name__ == "__main__":
    unittest.main()
