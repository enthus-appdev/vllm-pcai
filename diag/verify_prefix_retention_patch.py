"""CPU source-isolation checks for bounded SWA replay retention; no GPU imports."""
import ast
import pathlib
import sys
from types import SimpleNamespace


class SlidingWindowSpec(SimpleNamespace):
    pass


def load_function(tree, name, namespace, class_name=None):
    parent = tree if class_name is None else next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    node = next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<isolated source>", "exec"), namespace)
    return namespace[name]


def verify(root):
    root = pathlib.Path(root)
    coordinator = ast.parse((root / "v1/core/kv_cache_coordinator.py").read_text())
    managers = ast.parse((root / "v1/core/single_type_kv_cache_manager.py").read_text())
    ns = {"SlidingWindowSpec": SlidingWindowSpec, "cdiv": lambda a, b: (a + b - 1) // b,
          "resolve_block_hashes": lambda hashes, *args, **kwargs: hashes}
    replay = load_function(coordinator, "replay_boundaries", ns)
    cache = load_function(coordinator, "cache_blocks", ns, "KVCacheCoordinator")
    policy = next(n for n in ast.walk(coordinator) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Attribute) and t.attr == "keep_previous_swa_replay_boundary" for t in n.targets))
    expression = compile(ast.Expression(policy.value), "<policy>", "eval")
    for lookahead, eagle_groups, expected in [(0, set(), False), (1, set(), True),
                                              (1, {0}, False), (5, set(), False)]:
        assert eval(expression, {"num_prefill_lookahead": lookahead,
                                 "self": SimpleNamespace(eagle_group_ids=eagle_groups)}) == expected

    class Manager:
        supports_fine_grained_hash_lookup = False

    for name in ("_contiguous_blocks_for_hit", "reachable_block_mask", "find_longest_cache_hit"):
        setattr(Manager, name, classmethod(load_function(managers, name, ns, "SlidingWindowManager")))

    def boundaries(prompt, block, enabled=True, eagle=False):
        captured = []
        group_list = [SimpleNamespace(enable_caching=True, kv_cache_spec=spec,
                       cache_blocks=lambda *a, **kw: captured.append(kw["replay_boundaries"]))
                      for spec in (SlidingWindowSpec(), object())]
        self = SimpleNamespace(scheduler_block_size=block, eagle_group_ids={0} if eagle else set(),
                               keep_previous_swa_replay_boundary=enabled,
                               single_type_managers=group_list, retention_interval=0,
                               get_num_cacheable_tokens=lambda count, group: count)
        cache(self, SimpleNamespace(num_prompt_tokens=prompt), prompt)
        assert captured[1] == replay(prompt, block, eagle), "non-SWA retention changed"
        return captured[0]

    # Full residue sweep, including short prompts and larger hit alignment.
    cases = 0
    for block in (256, 1024):
        for prompt in [1, block, block + 1] + list(range(400 * block, 401 * block)):
            current = (prompt - 1) // block * block
            previous = max(0, current - block)
            kept = boundaries(prompt, block)
            assert prompt - 1 in kept and previous in kept
            assert len(kept) <= 2
            assert boundaries(prompt, block, False) == replay(prompt, block, False)
            assert boundaries(prompt, block, False, True) == replay(prompt, block, True)
            if previous == 0:
                continue
            physical_block = 256
            for window in (128, 512):
                spec = SlidingWindowSpec(block_size=physical_block, sliding_window=window)
                n = prompt // physical_block
                cold = list(range(n))
                # A changed suffix places the first mismatch in the last semantic block.
                lcp = prompt - 4
                changed = [i if (i + 1) * physical_block <= lcp else -(i + 1) for i in cold]
                for interval in (0, 4096, None):
                    masks = []
                    for enabled in (False, True):
                        mask = Manager.reachable_block_mask(0, n, block, spec, False,
                            interval, boundaries(prompt, block, enabled))
                        masks.append([True] * n if mask is None else mask)
                    assert all(not old or new for old, new in zip(*masks)), "lost a retained block"
                    assert sum(masks[1]) - sum(masks[0]) <= (window - 2) // physical_block + 1
                    if interval is None:
                        assert masks[0] == masks[1], "dense retention changed"
                    for hashes in (cold, changed):
                        hits = []
                        for mask in masks:
                            stored = {i for i, keep in enumerate(mask) if keep}
                            pool = SimpleNamespace(hash_block_size=physical_block, null_block=None,
                                get_cached_block=lambda h, gids: [h + 1] if h in stored else None)
                            _, hit = Manager.find_longest_cache_hit(hashes, prompt - 1, [0], pool, spec, False, block)
                            hits.append(hit)
                        assert hits[1] >= hits[0] and hits[1] > 0
                        if hashes is cold:
                            assert hits[1] == current, "exact-resend latest boundary lost"
                        elif interval == 0:
                            bad_residue = prompt % block in (1, 2, 3)
                            assert (hits[0] == 0) == bad_residue, "negative control no longer reproduces"
                            assert hits[1] == (previous if bad_residue else current)
                        cases += 1
    print(f"SWA prefix-retention CPU tripwire OK: {cases} lookup cases; exact/changed resends, bounded fallback, dense/periodic controls")


if __name__ == "__main__":
    verify(sys.argv[1])
