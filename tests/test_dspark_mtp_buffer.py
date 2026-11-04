"""CPU policy tests; GPU capture/replay and KV-budget validation are separate."""
import ast
import os
import pathlib
import types
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/dspark-unused-mtp-buffer.patch"


def load_policy():
    lines = PATCH.read_text().splitlines()
    start = next(i for i, line in enumerate(lines)
                 if line.startswith("+def _pcai_dspark_aux_only"))
    body = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        body.append(line[1:])
    tree = ast.parse("\n".join(body))
    fn = tree.body[0]
    fn.args.args[0].annotation = None
    fn.returns = None
    # Supply the existing resolver directly; importing vLLM requires CUDA deps.
    fn.body = [node for node in fn.body if not isinstance(node, ast.ImportFrom)]
    namespace = {"os": os, "get_eagle3_aux_layers_from_config": lambda s: s.layers}
    exec(compile(ast.fix_missing_locations(tree), str(PATCH), "exec"), namespace)
    return namespace["_pcai_dspark_aux_only"]


def config(method="dspark", layers=(41, 42, 43), v2=True, pp=1):
    ns = types.SimpleNamespace
    return ns(speculative_config=ns(use_dspark=lambda: method == "dspark", layers=layers),
              use_v2_model_runner=v2,
              parallel_config=ns(pipeline_parallel_size=pp),
              model_config=ns(hf_config=ns(num_hidden_layers=43)))


class DSparkBufferPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_default_and_non_one_values_preserve_buffer(self):
        for value in (None, "0", "true", ""):
            with patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ["VLLM_PCAI_DSPARK_SKIP_MTP_BUFFER"] = value
                self.assertFalse(self.policy(config()))

    def test_opt_in_aux_dspark_only(self):
        with patch.dict(os.environ, {"VLLM_PCAI_DSPARK_SKIP_MTP_BUFFER": "1"}):
            self.assertTrue(self.policy(config()))
            for method in ("mtp", "eagle", "eagle3", "dflash", "draft_model"):
                self.assertFalse(self.policy(config(method=method)))
            cfg = config()
            cfg.speculative_config = None
            self.assertFalse(self.policy(cfg))

    def test_missing_invalid_aux_keeps_fallback(self):
        with patch.dict(os.environ, {"VLLM_PCAI_DSPARK_SKIP_MTP_BUFFER": "1"}):
            for layers in (None, (), (0,), (44,), ("41",)):
                self.assertFalse(self.policy(config(layers=layers)))

    def test_v1_and_pipeline_parallel_keep_buffer(self):
        with patch.dict(os.environ, {"VLLM_PCAI_DSPARK_SKIP_MTP_BUFFER": "1"}):
            self.assertFalse(self.policy(config(v2=False)))
            self.assertFalse(self.policy(config(pp=2)))

    def test_patch_only_changes_constructor_allocation_policy(self):
        text = PATCH.read_text()
        self.assertNotIn("\n-", text.replace("--- a/", "baseline "))
        self.assertIn("+            needs_mtp_hidden_states = False", text)
        self.assertEqual(text.count("diff --git "), 1)
        # No capture, replay, aux arithmetic, or forward-copy changes.
        for expression in ("copy_(", "combine_hidden_states(", "use_workspace_lane("):
            self.assertNotIn("+" + expression, text)


if __name__ == "__main__":
    unittest.main()
