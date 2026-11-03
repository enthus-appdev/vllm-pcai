"""Catch malformed patch files before downloading the base image."""

import hashlib
import pathlib
import re
import subprocess
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class PatchIntegrityTests(unittest.TestCase):
    def test_audited_base_and_patch_inventory(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(
            "FROM vllm/vllm-openai:v0.33.0@sha256:"
            "d825baf9ce4b1b6bfc5ecc7b79880c7e6748c65a72d697ce3d47607aafa84c6e",
            dockerfile,
        )
        copied = re.findall(r"^COPY patches/(\S+) ", dockerfile, re.MULTILINE)
        self.assertEqual(set(copied), {p.name for p in (ROOT / "patches").glob("*.patch")})
        self.assertEqual(len(copied), len(set(copied)), "patch applied twice")

    def test_shm_fix_is_not_patched(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertNotIn("COPY patches/63165-", dockerfile)

    def test_pinned_c128_runtime_patch(self):
        patch = ROOT / "patches" / "58074-deepseek-v4-c128-circular-state-v033.patch"
        self.assertEqual(
            hashlib.sha256(patch.read_bytes()).hexdigest(),
            "ac31e8167e3410e64d9177a7b83e2fda2dffc4090bb53f35ba23a54e1c99bfae",
        )
        paths = {
            line.split()[2].removeprefix("a/")
            for line in patch.read_text().splitlines()
            if line.startswith("diff --git ")
        }
        self.assertEqual(
            paths,
            set(
                [
                    "vllm/models/deepseek_v4/common/ops/save_partial_states.py",
                    "vllm/models/deepseek_v4/compressor.py",
                    "vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py",
                ]
            ),
        )
        self.assertTrue(all(path.endswith(".py") for path in paths))
        self.assertNotIn("csrc/", patch.read_text())
        self.assertIn(
            hashlib.sha256(patch.read_bytes()).hexdigest(),
            (ROOT / "Dockerfile").read_text(),
        )
        self.assertIn(f"COPY patches/{patch.name} ", (ROOT / "Dockerfile").read_text())

    def test_pinned_swa_admission_patch(self):
        patch = ROOT / "patches" / "60045-swa-admission-held-blocks.patch"
        digest = "d2e0787a02594e4582ebdad78c92d928fd3f19dfc3bed6d98b192f6cc4c729a4"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {
            line.split()[2].removeprefix("a/")
            for line in patch.read_text().splitlines()
            if line.startswith("diff --git ")
        }
        self.assertEqual(paths, {"vllm/v1/core/single_type_kv_cache_manager.py"})
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)

    def test_pinned_stability_bundle(self):
        patch = ROOT / "patches/63427-nested-weak-ref-tensors-v033.patch"
        digest = "98d005b2c87c9942b01257ce7f2105519b616f7b7ecfc4016610c6e30ae7849d"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {line.split()[2].removeprefix("a/") for line in patch.read_text().splitlines()
                 if line.startswith("diff --git ")}
        self.assertEqual(paths, {"vllm/utils/torch_utils.py"})
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)

    def test_fp8_uses_upstream_code_and_dependencies(self):
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertNotIn("COPY patches/vision-fp8", dockerfile)
        self.assertNotIn("pip install", dockerfile)

    def test_pinned_prefix_retention_fallback(self):
        patch = ROOT / "patches/dspark-swa-prefix-fallback-v033.patch"
        digest = "83e748d8821e8a715a3fe3acd2e0cb0e43f6b75ed60853853a9e5a29e0400512"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {line.split()[2].removeprefix("a/") for line in patch.read_text().splitlines()
                 if line.startswith("diff --git ")}
        self.assertEqual(paths, {"vllm/v1/core/kv_cache_coordinator.py"})
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)
        self.assertNotIn("ENV VLLM_PREFIX_CACHE_RETENTION", dockerfile)

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

    def test_pinned_load_order_patch(self):
        patch = ROOT / "patches" / "60078-simple-offload-load-after-compute.patch"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), "6efcda2d6eead265dca1c14b1c27175809b5e523226ff15ee10860c4fcd624dc")
        self.assertIn("6efcda2d6eead265dca1c14b1c27175809b5e523226ff15ee10860c4fcd624dc", (ROOT / "Dockerfile").read_text())
        self.assertIn(f"COPY patches/{patch.name} ", (ROOT / "Dockerfile").read_text())
        paths = {line.split()[2].removeprefix("a/") for line in patch.read_text().splitlines()
                 if line.startswith("diff --git ")}
        self.assertEqual(paths, {"vllm/v1/simple_kv_offload/worker.py"})

    def test_embed_offload_plugin_layout(self):
        root = ROOT / "plugins" / "pcai_embed_offload"
        self.assertTrue((root / "pcai_embed_offload.py").is_file())
        eps = (root / "pcai_embed_offload-1.0.dist-info" / "entry_points.txt").read_text()
        self.assertIn("pcai_embed_offload = pcai_embed_offload:register", eps)
        self.assertIn("COPY plugins/pcai_embed_offload ", (ROOT / "Dockerfile").read_text())


if __name__ == "__main__":
    unittest.main()
