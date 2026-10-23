"""Catch malformed patch files before downloading the base image."""

import hashlib
import pathlib
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PatchIntegrityTests(unittest.TestCase):
    def test_pinned_runtime_patch(self):
        patch = ROOT / "patches" / "63165-shm-notify-backlog.patch"
        self.assertEqual(
            hashlib.sha256(patch.read_bytes()).hexdigest(),
            "468f5d097ad180363cc8133021b96ee7fc16aba1b48e1631a194be79f84ea609",
        )
        paths = {
            line.split()[2].removeprefix("a/")
            for line in patch.read_text().splitlines()
            if line.startswith("diff --git ")
        }
        self.assertEqual(
            paths, set(["vllm/distributed/device_communicators/shm_broadcast.py"])
        )
        self.assertTrue(all(path.endswith(".py") for path in paths))
        self.assertNotIn("csrc/", patch.read_text())
        self.assertIn(
            hashlib.sha256(patch.read_bytes()).hexdigest(),
            (ROOT / "Dockerfile").read_text(),
        )
        self.assertIn(f"COPY patches/{patch.name} ", (ROOT / "Dockerfile").read_text())

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
