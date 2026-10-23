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

    def test_fixed_width_patch_is_pinned_and_python_only(self):
        patch = ROOT / "patches" / "58068-fixed-width-indexer-logits-v0.32.patch"
        digest = "967eda4bd76a3f94cbdfccaebec7e9bbd27b86cb8be225ab5c2b666924cffb86"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {
            line.split()[2].removeprefix("a/")
            for line in patch.read_text().splitlines()
            if line.startswith("diff --git ")
        }
        self.assertEqual(
            paths,
            {
                "vllm/envs.py",
                "vllm/model_executor/layers/sparse_attn_indexer.py",
                "vllm/models/glm5next/nvidia/sparse_indexer.py",
                "vllm/utils/deep_gemm.py",
                "vllm/v1/attention/backends/mla/indexer.py",
            },
        )
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)
        self.assertNotIn("ENV VLLM_SPARSE_INDEXER_FIXED_LOGITS_WIDTH", dockerfile)

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
