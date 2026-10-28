"""Bounded two-GPU synthetic smoke: no model load, no endpoint, no benchmark."""
from datetime import timedelta
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


def native_smoke():
    import vllm_exl3_c as ext
    torch.manual_seed(719)
    for bits in (2, 3):
        inputs, outputs = 1024, 2048
        codes = torch.randint(-32768, 32767, (inputs // 16, outputs // 16, 16 * bits),
                              dtype=torch.int16, device='cuda')
        suh = torch.ones(inputs, dtype=torch.float16, device='cuda')
        svh = torch.ones(outputs, dtype=torch.float16, device='cuda')
        weights = ext.dequant_trellis(codes, suh, svh, bits, True)
        for tokens in (1, 4):
            x = torch.randn(tokens, inputs, dtype=torch.float16, device='cuda') * 0.1
            reference = x.float() @ weights.float()
            actual = ext.exl3_gemv(x, codes, suh, svh, bits, True)
            torch.cuda.synchronize()
            error = (actual.float() - reference).norm() / reference.norm()
            if not torch.isfinite(actual).all() or error.item() >= 0.05:
                raise AssertionError(f'native K{bits} GEMV parity error {error.item()}')
            print(json.dumps({'rank': dist.get_rank(), 'native_bits': bits,
                              'tokens': tokens, 'relative_error': error.item()}), flush=True)


def write_fixture(root, heads):
    rows, dim = sum(heads), 256
    values = bytes([0x38 + row % 3 for row in range(rows) for _ in range(dim)])
    scales = bytes([125 + row % 5 for row in range(rows) for _ in range(dim // 32)])
    key = 'layers.1.engram.embed'
    meta = {
        key + '.weight': dict(dtype='F8_E4M3', shape=[rows, dim], data_offsets=[0, len(values)]),
        key + '.scale': dict(dtype='U8', shape=[rows, dim // 32],
                             data_offsets=[len(values), len(values) + len(scales)]),
    }
    header = json.dumps(meta).encode()
    (root / 'table.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + values + scales)
    (root / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {k: 'table.safetensors' for k in meta}}))


def engram_smoke(root, heads):
    from pcai_eaq.runtime import install
    from vllm.models.deepseek_v41.common import engram as impl
    if dist.get_rank() == 0:
        root.mkdir(parents=True, exist_ok=True)
        write_fixture(root, heads)
    dist.barrier()
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        engram_config=SimpleNamespace(cpu_offload=False, dp_shared_memory=False, use_thp=False),
    )
    # Only the adapter's constructor config is synthetic. vLLM TP/NCCL is real.
    with patch('vllm.config.get_current_vllm_config', lambda: cfg):
        install()
    shell = object.__new__(impl.Engram)
    torch.nn.Module.__init__(shell)
    shell._pcai_table_prefix = 'model.layers.1.engram'
    layout = SimpleNamespace(head_dim=256, primes=[(heads,)], num_embeddings=[sum(heads)])
    os.environ['PCAI_ENGRAM_DIR'] = str(root)
    table = shell._create_embedding(layout, 0)
    try:
        if list(table.parameters()):
            raise AssertionError('disk adapter must not allocate table Parameters')
        start = [sum(heads[:i]) for i in range(len(heads))]
        ids = torch.tensor([[s + 1 for s in start], [-1] + [s + 2 for s in start[1:]]],
                           dtype=torch.int64, device='cuda')
        rows, dim = sum(heads), 256
        weights = torch.tensor([0x38 + row % 3 for row in range(rows) for _ in range(dim)],
                               dtype=torch.uint8, device='cuda').view(torch.float8_e4m3fn).view(rows, dim)
        scales = torch.tensor([125 + row % 5 for row in range(rows) for _ in range(dim // 32)],
                              dtype=torch.uint8, device='cuda').view(rows, dim // 32)

        def native_lookup(head_start, count):
            out = torch.empty((2, count, dim), dtype=torch.bfloat16, device='cuda')
            first, end = sum(heads[:head_start]), sum(heads[:head_start + count])
            # Original kernel addresses local contiguous table slices, not global row zero.
            impl._engram_lookup_kernel[(1,)](
                weights[first:end], scales[first:end], ids, ids, out,
                first, end, 2 * count, ids.stride(0), ids.stride(1),
                HEAD_START=head_start, LOCAL_HEADS=count, TOTAL_HEADS=len(heads),
                DIM=dim, QUANT_BLOCK=32, BLOCK_R=16, GRID=1,
                SORTED=False, PACKED=False, OUT_STRIDE=out.stride(0))
            return out

        local = torch.empty((2, table.part_n_hash_cols, dim), dtype=torch.bfloat16, device='cuda')
        table.lookup(ids, local)
        reference = native_lookup(table.head_start, table.part_n_hash_cols)
        torch.testing.assert_close(local, reference, rtol=0, atol=0)
        gathered = table.forward(ids)
        torch.testing.assert_close(gathered, native_lookup(0, len(heads)), rtol=0, atol=0)
        torch.cuda.synchronize()
        print(json.dumps({'rank': dist.get_rank(), 'engram_heads': len(heads),
                          'head_start': table.head_start, 'tp_native_parity': 'PASS'}), flush=True)
    finally:
        table.values.close()
        table.scales.close()
    dist.barrier()


def main():
    rank = int(os.environ['LOCAL_RANK'])
    if torch.cuda.device_count() != 2:
        raise RuntimeError('synthetic qualification requires exactly two visible GPUs')
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability(rank) != (9, 0):
        raise RuntimeError('this fixture is specifically for H100/sm90')
    from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment, initialize_model_parallel,
        destroy_model_parallel, destroy_distributed_environment,
    )
    # vLLM group initialization needs a real config context even without a model.
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))
    with set_current_vllm_config(config):
        try:
            init_distributed_environment(world_size=2, rank=rank, local_rank=rank,
                                         backend='nccl', timeout=timedelta(seconds=180))
            initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1,
                                      backend='nccl')
            native_smoke()
            engram_smoke(Path('/fixtures/even'), (11, 13, 17, 19))
            engram_smoke(Path('/fixtures/padded'), (11, 13, 17))
            print(json.dumps({'rank': rank, 'synthetic_smoke': 'PASS',
                              'full_model_qualified': False}), flush=True)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


if __name__ == '__main__':
    main()
