"""Pin the vendored vllm#64177 backport and its Dockerfile application."""

import hashlib
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "64177-omit-null-usage-details.patch"
DIGEST = "c558a2898b0eb96454e2df76c4e3169506d120d8b471e04f98b1a4f77aa65416"


class Backport64177Tests(unittest.TestCase):
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
