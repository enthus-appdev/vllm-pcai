"""CPU source-isolation checks for the opt-in fixed-width logits backport."""

import ast
import os
import pathlib
import types
from unittest.mock import patch


def execute(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, "fixed-width-tripwire", "exec"), namespace)


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


def verify(source_root: pathlib.Path) -> None:
    env_tree = ast.parse((source_root / "envs.py").read_text())
    flag_name = "VLLM_SPARSE_INDEXER_FIXED_LOGITS_WIDTH"
    env_dict = next(
        node
        for node in ast.walk(env_tree)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(key, ast.Constant) and key.value == flag_name
            for key in node.keys
        )
    )
    flag = next(
        value
        for key, value in zip(env_dict.keys, env_dict.values)
        if isinstance(key, ast.Constant) and key.value == flag_name
    )
    reader = eval(
        compile(ast.Expression(body=flag), "env-tripwire", "eval"), {"os": os}
    )
    for raw, expected in ((None, None), ("0", False), ("1", True)):
        with patch.dict(
            os.environ, {} if raw is None else {flag_name: raw}, clear=True
        ):
            assert reader() is expected

    tree = ast.parse((source_root / "v1/attention/backends/mla/indexer.py").read_text())
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepseekV32IndexerMetadataBuilder"
    )
    planner = next(
        node
        for node in builder.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_split_indexer_prefill_chunks"
    )
    planner.decorator_list = []
    namespace = {}
    execute([planner], namespace)
    split = namespace[planner.name]
    width, budget, query_rows = 196608, 128 * 2**20, 4096
    fixed_plans = []
    for keys in (1024, 16384, width):
        args = ([Scalar(keys)], [Scalar(query_rows)], 4 * 786432, budget)
        chunks = split(*args, logits_width=width)
        heights = [query.stop - query.start for _, query in chunks]
        fixed_plans.append(heights)
        assert sum(heights) == query_rows
        assert all(height * width * 4 <= budget for height in heights)
        assert split(*args, logits_width=0) == split(*args)
    assert fixed_plans[0] == fixed_plans[1] == fixed_plans[2]
    assert len(fixed_plans[0]) == 25 and max(fixed_plans[0]) == 170
    chunks = split(
        [Scalar(262144)], [Scalar(2048)], 4 * 1048576, budget, logits_width=262144
    )
    assert len(chunks) == 16 and all(
        query.stop - query.start == 128 for _, query in chunks
    )
    chunks = split(
        [Scalar(100), Scalar(100)],
        [Scalar(1000), Scalar(1000)],
        4 * 786432,
        budget,
        logits_width=width,
    )
    assert sum(query.stop - query.start for _, query in chunks) == 2000
    assert all((query.stop - query.start) * width * 4 <= budget for _, query in chunks)
    chunks = split(
        [Scalar(width * 2)],
        [Scalar(query_rows)],
        4 * 786432,
        budget,
        logits_width=width,
    )
    assert all(
        (query.stop - query.start) * (width * 2) * 4 <= budget for _, query in chunks
    )

    init = next(
        node
        for node in builder.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    start = next(
        index
        for index, node in enumerate(init.body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "use_fixed_width" for t in node.targets
        )
    )
    selection = init.body[start : start + 4]
    for override, integrated, dcp, expected in (
        (None, False, 1, 0),
        (None, True, 1, width),
        (False, True, 1, 0),
        (True, False, 1, width),
        (True, False, 2, 0),
    ):
        state = types.SimpleNamespace(
            dcp_world_size=dcp,
            compress_ratio=4,
            vllm_config=types.SimpleNamespace(
                model_config=types.SimpleNamespace(max_model_len=786432)
            ),
        )
        execute(
            selection,
            {
                "self": state,
                "envs": types.SimpleNamespace(**{flag_name: override}),
                "current_platform": types.SimpleNamespace(
                    is_integrated_gpu=lambda: integrated
                ),
            },
        )
        assert state.fixed_logits_width == expected

    deepgemm = ast.parse((source_root / "utils/deep_gemm.py").read_text())
    wrapper = next(
        node
        for node in deepgemm.body
        if isinstance(node, ast.FunctionDef) and node.name == "fp8_fp4_mqa_logits"
    )
    calls = []

    def implementation(*args, **kwargs):
        calls.append(kwargs)
        return "sentinel"

    ns = {"_lazy_init": lambda: None, "_fp8_fp4_mqa_logits_impl": implementation}
    execute([wrapper], ns)
    assert (
        ns[wrapper.name](None, None, None, None, None, False, max_seqlen_k=width)
        == "sentinel"
    )
    assert calls[-1] == {"clean_logits": False, "max_seqlen_k": width}

    for path in (
        "model_executor/layers/sparse_attn_indexer.py",
        "models/glm5next/nvidia/sparse_indexer.py",
    ):
        runtime = ast.parse((source_root / path).read_text())
        assert any(
            isinstance(node, ast.Call)
            and any(keyword.arg == "max_seqlen_k" for keyword in node.keywords)
            for node in ast.walk(runtime)
        )


if __name__ == "__main__":
    import sys

    verify(pathlib.Path(sys.argv[1]))
    print(
        "Fixed-width env opt-in, DCP guard, chunk budgets and DeepGEMM keyword forwarding OK (CPU source isolation)"
    )
