"""CPU arithmetic/TP fixture tests with a fake vLLM graph, not GPU qualification."""
import json
import os
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcai_eaq.runtime import install


def table_file(root, heads):
    rows = sum(heads)
    values = bytes(0x38 + row % 3 for row in range(rows) for _ in range(32))
    scales = bytes([127] * rows)
    prefix = 'layers.1.engram.embed'
    meta = {
        prefix + '.weight': {'dtype': 'F8_E4M3', 'shape': [rows, 32],
                             'data_offsets': [0, len(values)]},
        prefix + '.scale': {'dtype': 'U8', 'shape': [rows, 1],
                            'data_offsets': [len(values), len(values) + len(scales)]},
    }
    header = json.dumps(meta).encode()
    (root / 'table.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + values + scales)
    (root / 'model.safetensors.index.json').write_text(json.dumps(
        {'weight_map': {key: 'table.safetensors' for key in meta}}))


def fixture(root, heads, rank, eager=True):
    cfg = types.SimpleNamespace(
        model_config=types.SimpleNamespace(enforce_eager=eager),
        parallel_config=types.SimpleNamespace(data_parallel_size=1),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=2),
        engram_config=types.SimpleNamespace(cpu_offload=False, dp_shared_memory=False, use_thp=False),
    )
    ready = False

    def current():
        if not ready:
            raise AssertionError('configuration requested during plugin registration')
        return cfg

    class Engram(torch.nn.Module):
        def __init__(self, config, quant_config, layout, layer_hash_index,
                     use_sequence_parallel, prefix, prefetch_stream=None):
            super().__init__()
            self.embed_tokens = self._create_embedding(layout, layer_hash_index)

    fake = {
        'vllm': types.ModuleType('vllm'),
        'vllm.config': types.SimpleNamespace(get_current_vllm_config=current),
        'vllm.distributed': types.SimpleNamespace(
            get_tensor_model_parallel_rank=lambda: rank,
            get_tensor_model_parallel_world_size=lambda: 2,
            tensor_model_parallel_all_gather=lambda out, dim: out,
        ),
        'vllm.models': types.ModuleType('vllm.models'),
        'vllm.models.deepseek_v41': types.ModuleType('vllm.models.deepseek_v41'),
        'vllm.models.deepseek_v41.common': types.SimpleNamespace(
            engram=types.SimpleNamespace(Engram=Engram)),
    }
    layout = types.SimpleNamespace(head_dim=32, primes=[(heads,)], num_embeddings=[sum(heads)])
    with patch.dict(sys.modules, fake), patch.dict(os.environ, PCAI_ENGRAM_DIR=str(root)):
        install()
        install()  # idempotent
        ready = True
        model = Engram(None, None, layout, 0, False, 'model.layers.1.engram')
    return model.embed_tokens


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        table_file(self.root, (11, 13, 17, 19))
        self.capture = patch.object(torch.cuda, 'is_current_stream_capturing', return_value=False)
        self.capture.start()

    def tearDown(self):
        self.capture.stop()
        self.tmp.cleanup()

    def close(self, embedding):
        embedding.values.close()
        embedding.scales.close()

    def test_tp2_rows_and_no_parameters(self):
        ids = torch.tensor([[1, 12, 25, 42], [-1, 13, 26, 43]])
        for rank in [0, 1]:
            emb = fixture(self.root, (11, 13, 17, 19), rank)
            try:
                self.assertEqual(list(emb.parameters()), [])
                out = torch.empty(2, 2, 32, dtype=torch.bfloat16)
                emb.lookup(ids, out)
                for t in range(2):
                    for h in range(2):
                        row = ids[t, rank * 2 + h].item()
                        expected = 0.0 if row == -1 else 1 + (row % 3) / 8
                        self.assertTrue(torch.all(out[t, h] == expected))
            finally:
                self.close(emb)

    def test_padded_head_is_zero(self):
        table_file(self.root, (11, 13, 17))
        emb = fixture(self.root, (11, 13, 17), 1)
        try:
            out = torch.empty(1, 2, 32, dtype=torch.bfloat16)
            emb.lookup(torch.tensor([[1, 12, 25]]), out)
            self.assertTrue(torch.all(out[:, 1] == 0))
        finally:
            self.close(emb)

    def test_reject_graph_config(self):
        with self.assertRaisesRegex(RuntimeError, 'enforce-eager'):
            fixture(self.root, (11, 13, 17, 19), 0, eager=False)

    def test_reject_foreign_row_and_capacity(self):
        emb = fixture(self.root, (11, 13, 17, 19), 0)
        try:
            with self.assertRaisesRegex(ValueError, 'another TP shard'):
                emb.lookup(torch.tensor([[30, 12, 25, 42]]), torch.empty(1, 2, 32, dtype=torch.bfloat16))
            with self.assertRaisesRegex(ValueError, 'budget'):
                emb.lookup(torch.zeros(3, 4, dtype=torch.long), torch.empty(3, 2, 32, dtype=torch.bfloat16))
        finally:
            self.close(emb)


if __name__ == '__main__':
    unittest.main()
