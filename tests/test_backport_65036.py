"""Pin the vendored vllm#65036 backport and its Dockerfile application."""

import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "65036-stream-usage-cache-split.patch"
DIGEST = "c8097e10297d8a964c7a3588885002a657db0309d7734343ed6a3e011ef64a07"


class Backport65036Tests(unittest.TestCase):
    def test_patch_is_pinned_and_applied(self):
        self.assertEqual(hashlib.sha256(PATCH.read_bytes()).hexdigest(), DIGEST)
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(DIGEST, dockerfile)
        self.assertIn(f"COPY patches/{PATCH.name} ", dockerfile)

    def test_patch_vendors_production_files_only(self):
        paths = [line.split()[2] for line in PATCH.read_text().splitlines()
                 if line.startswith("diff --git ")]
        self.assertTrue(paths)
        self.assertTrue(all(path.startswith("a/vllm/") for path in paths), paths)


if __name__ == "__main__":
    unittest.main()
