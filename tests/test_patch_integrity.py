"""Catch malformed patch files before downloading the base image."""

import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PatchIntegrityTests(unittest.TestCase):
    def test_git_can_parse_every_patch(self):
        patches = sorted((ROOT / "patches").glob("*.patch"))
        self.assertTrue(patches, "No vendored patches found")
        for patch in patches:
            with self.subTest(patch=patch.name):
                result = subprocess.run(
                    ["git", "apply", "--numstat", str(patch)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip(), "Patch contains no file changes")


if __name__ == "__main__":
    unittest.main()
