"""Pin the vendored vllm#65271 backport and its Dockerfile application."""

import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "65271-mm-prefix-split.patch"
DIGEST = "8bfa08f8b8e72762f9b976f4adfccf3cd881ea7ee71bf020a15273853ce14c89"


class Backport65271Tests(unittest.TestCase):
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
