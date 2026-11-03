"""Pin the vendored vllm#64345 backport and its Dockerfile application."""

import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "64345-host-pool-no-zero-fill.patch"
DIGEST = "35efc65ae158e1fafb7f59e799b26fcee62f949bd7164752702d1c8cd9081989"


class Backport64345Tests(unittest.TestCase):
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
