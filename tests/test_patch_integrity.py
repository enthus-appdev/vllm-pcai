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

    def test_pinned_c128_runtime_patch(self):
        patch = ROOT / "patches" / "58074-deepseek-v4-c128-circular-state.patch"
        self.assertEqual(
            hashlib.sha256(patch.read_bytes()).hexdigest(),
            "cd299bb8fe6f1e0501d4fa2b63742f08f91549206e88cb6de3b795ffbcb0f9d0",
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

    def test_pinned_simple_offload_lru_patch(self):
        patch = ROOT / "patches" / "62444-simple-offload-lru-refresh.patch"
        digest = "022cf9e9a8d34b6a9330e472ee2fcc887a6c07643eee4b3352d4b1b556d13970"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {
            line.split()[2].removeprefix("a/")
            for line in patch.read_text().splitlines()
            if line.startswith("diff --git ")
        }
        self.assertEqual(paths, {"vllm/v1/simple_kv_offload/manager.py"})
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)
        self.assertIn("python3 /tmp/verify_simple_offload_lru_patch.py", dockerfile)

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
        self.assertIn("python3 /tmp/verify_swa_admission_patch.py", dockerfile)

    def test_pinned_stability_bundle(self):
        patch = ROOT / "patches/vision-stability-v032.patch"
        digest = "185ef3b32eab7aefab857e81de107b5714ed94525df654712e1616753f2b7668"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {line.split()[2].removeprefix("a/") for line in patch.read_text().splitlines()
                 if line.startswith("diff --git ")}
        self.assertEqual(paths, {"vllm/config/speculative.py", "vllm/utils/torch_utils.py",
                                 "vllm/v1/core/sched/scheduler.py"})
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)
        self.assertIn("python3 /tmp/verify_stability_patch.py", dockerfile)

    def test_pinned_fp8_moe_bundle(self):
        patch = ROOT / "patches/vision-fp8-moe-v032.patch"
        digest = "ffa8397c9ddea324e74f4ef259d3d799a5425eca69f56e08a8125683cdbe31d2"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {line.split()[2].removeprefix("a/") for line in patch.read_text().splitlines()
                 if line.startswith("diff --git ")}
        self.assertEqual(paths, {"vllm/model_executor/layers/fused_moe/oracle/mxfp4.py",
                                 "vllm/models/deepseek_v4/quant_config.py",
                                 "vllm/models/deepseek_v41/quant_config.py", "vllm/utils/humming.py"})
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)
        self.assertIn("python3 /tmp/verify_fp8_moe_patch.py", dockerfile)
        self.assertIn("humming-kernels[cu13]==0.1.18", dockerfile)
        self.assertIn("--no-deps --only-binary=:all: --require-hashes", dockerfile)
        self.assertIn("3eee5f7365457708a8962c852a4c797d99439a450dfe16fee47cc39664d080d4", dockerfile)

    def test_pinned_indexer_memory_bundle(self):
        patch = ROOT / "patches/vision-indexer-memory-v032.patch"
        digest = "1fe6a6db9bf855caa3e27c8f67f829b49e203bf6c9b7995a79ebc45885feff36"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {line.split()[2].removeprefix("a/") for line in patch.read_text().splitlines()
                 if line.startswith("diff --git ")}
        self.assertEqual(paths, {"vllm/envs.py", "vllm/model_executor/layers/sparse_attn_indexer.py",
                                 "vllm/models/glm5next/nvidia/sparse_indexer.py",
                                 "vllm/utils/deep_gemm.py", "vllm/v1/attention/backends/mla/indexer.py"})
        self.assertNotIn("vllm/models/deepseek_v41/", patch.read_text())
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)
        self.assertIn("python3 /tmp/verify_indexer_memory_patch.py", dockerfile)
        self.assertIn("COPY diag/verify_fixed_width_logits_patch.py ", dockerfile)
        self.assertIn("ENV VLLM_SPARSE_INDEXER_FIXED_LOGITS_WIDTH=1", dockerfile)

    def test_pinned_dspark_int8_bundle(self):
        patch = ROOT / "patches/62958-dspark-int8-v032.patch"
        digest = "ffcfb0cc255452da8d5fe508efc08b1e7a4833820528a3cac2928c73e202e09a"
        self.assertEqual(hashlib.sha256(patch.read_bytes()).hexdigest(), digest)
        paths = {line.split()[2].removeprefix("a/") for line in patch.read_text().splitlines()
                 if line.startswith("diff --git ")}
        self.assertEqual(paths, {"csrc/dspark_markov_walk.cu", "vllm/_custom_ops.py",
                                 "vllm/config/speculative.py",
                                 "vllm/v1/worker/gpu/spec_decode/dspark/markov_walk.py",
                                 "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"})
        self.assertIn("Copyright (c) 2026 tie-pilot-qxw", patch.read_text())
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(digest, dockerfile)
        self.assertIn(f"COPY patches/{patch.name} ", dockerfile)
        self.assertIn("python3 /tmp/verify_dspark_int8_patch.py", dockerfile)
        self.assertIn("cuobjdump --list-elf", dockerfile)
        self.assertIn('patch --batch --forward --fuzz=0 -p1 -d "$SITE" < /tmp/dspark-int8.patch', dockerfile)
        self.assertNotIn("apply --include=", dockerfile)
        builder = (ROOT / "native/dspark/build_extension.py").read_text()
        self.assertIn("compute_90a,code=sm_90a", builder)
        self.assertIn("0x020B000000000000ULL", builder)
        self.assertIn('("USE_CUDA", "1")', builder)
        for path, expected in (
            ("torch_utils.h", "f619e1943039a67a3254f7e0679e1f3e0faa9d8fccae795f3b20df9e91a8a429"),
            ("core/registration.h", "52cc0c3cd375d5cca437cbc011519814aefe5fe55eb9698147da50b7a19314c8"),
        ):
            self.assertEqual(hashlib.sha256((ROOT / "native/dspark/include" / path).read_bytes()).hexdigest(), expected)
            self.assertIn(expected, dockerfile)

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
