"""CPU source-isolation checks; fake allocations never allocate tensors."""
import ast
import pathlib
import sys
import types

from verify_fixed_width_logits_patch import execute, verify as verify_fixed_width


def verify(root):
    root = pathlib.Path(root)
    verify_fixed_width(root)
    tree = ast.parse((root / "v1/attention/backends/mla/indexer.py").read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepseekV32IndexerMetadataBuilder")
    init = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    start = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "max_decode_tokens" for t in n.targets))
    for seqs, tokens, next_n, expected in ((8, 8192, 6, 48), (8, 16, 6, 16), (1, 8192, 1, 1), (64, 8192, 9, 576)):
        state = types.SimpleNamespace(device="cpu")
        fake_torch = types.SimpleNamespace(int32="int32", zeros=lambda shape, **kwargs: types.SimpleNamespace(shape=shape))
        execute(init.body[start:start + 2], {"self": state, "torch": fake_torch,
                "scheduler_config": types.SimpleNamespace(max_num_seqs=seqs, max_num_batched_tokens=tokens),
                "next_n": next_n, "block_table_width": 1024})
        assert state.expanded_block_table_buffer.shape == (expected, 1024)
    assert "assert num_decode_tokens <= self.expanded_block_table_buffer.shape[0]" in ast.unparse(builder)

    runtime = ast.parse((root / "model_executor/layers/sparse_attn_indexer.py").read_text())
    chunk_loop = next(n for n in ast.walk(runtime) if isinstance(n, ast.For) and ast.unparse(n.iter) == "prefill_metadata.chunks")
    assert isinstance(chunk_loop.body[-1], ast.Delete) and ast.unparse(chunk_loop.body[-1]) == "del logits", "chunk logits retained into next allocation"
    assert "_merge_dcp_topk_global(" in ast.unparse(chunk_loop), "DCP merge must run before freeing logits"
    # Execute only the profile allocation sequence, with inert allocation stand-ins.
    profile = next(n for n in ast.walk(runtime) if isinstance(n, ast.If) and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "max_logits_bytes" for t in x.targets) for x in n.body))
    start = next(i for i, n in enumerate(profile.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "max_logits_bytes" for t in n.targets))
    for cap, seqs, k, context, budget, expected in (
        (90, 8, 5, 262144, 512, 512 * 2**20),
        (90, 256, 5, 1048576, 512, 256 * 6 * 1048576 * 4),
        (90, 1, 0, 257, 0, 512 * 4),
        (100, 256, 5, 1048576, 512, 512 * 2**20),
    ):
        allocations = []
        ns = {"envs": types.SimpleNamespace(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=budget),
              "current_platform": types.SimpleNamespace(is_device_capability_family=lambda x: x == cap),
              "get_current_vllm_config_or_none": lambda: types.SimpleNamespace(
                  scheduler_config=types.SimpleNamespace(max_num_seqs=seqs, max_num_batched_tokens=8192), num_speculative_tokens=k),
              "max_model_len": context, "round_up": lambda x, n: -(-x // n) * n,
              "torch": types.SimpleNamespace(uint8="uint8", empty=lambda size, **kwargs: allocations.append(size)),
              "hidden_states": types.SimpleNamespace(device="cpu")}
        execute(profile.body[start:start + 3], ns)
        assert allocations == [expected], (cap, allocations, expected)
    print("indexer CPU tripwire: fixed-width budgets, bounded decode table, SM90 profile reserve and chunk lifetime OK")


if __name__ == "__main__":
    verify(sys.argv[1])
