"""Eager DP1/TP-sharded Engram adapter. GPU qualification is still required."""
import functools
import json
import os

from .rows import DirectTensor, head_partition, safe_path


def install():
    import torch
    from torch import nn
    from vllm.config import get_current_vllm_config
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_gather,
    )
    from vllm.models.deepseek_v41.common import engram as impl

    if getattr(impl.Engram, "_pcai_disk_installed", False):
        return
    # Plugin registration precedes current-config context; defer checks to construction.

    class DiskEmbedding(nn.Module):
        def __init__(self, layout, hash_index, prefix):
            super().__init__()
            cfg = get_current_vllm_config()
            if not cfg.model_config.enforce_eager:
                raise RuntimeError("disk Engram qualification requires --enforce-eager")
            if cfg.parallel_config.data_parallel_size != 1:
                raise RuntimeError("disk Engram qualification requires DP1")
            if cfg.engram_config and (
                cfg.engram_config.cpu_offload
                or cfg.engram_config.dp_shared_memory
                or cfg.engram_config.use_thp
            ):
                raise RuntimeError("disable stock Engram host/shared/THP placement")
            self.tp_size = get_tensor_model_parallel_world_size()
            self.dp_size = 1
            self.cpu_offload = False  # Keep Engram staging on the synchronous eager path.
            self.dp_shared_memory = False
            self.dim = layout.head_dim
            self.block_size = 32
            if self.dim % self.block_size:
                raise ValueError("invalid quant block width")
            heads = tuple(n for order in layout.primes[hash_index] for n in order)
            self.n_hash_cols = len(heads)
            (self.head_start, self.part_n_hash_cols, self.vocab_start_idx,
             self.vocab_end_idx) = head_partition(
                 heads, get_tensor_model_parallel_rank(), self.tp_size)
            self.num_embeddings = layout.num_embeddings[hash_index]
            self.part_num_embeddings = self.vocab_end_idx - self.vocab_start_idx
            self.max_tokens = cfg.scheduler_config.max_num_batched_tokens
            root = os.environ["PCAI_ENGRAM_DIR"]
            index = json.loads(safe_path(root, "model.safetensors.index.json").read_text())["weight_map"]
            raw = prefix.removeprefix("language_model.").removeprefix("model.") + ".embed"
            weight_key, scale_key = raw + ".weight", raw + ".scale"
            self.values = DirectTensor(root, index[weight_key], weight_key)
            self.scales = DirectTensor(root, index[scale_key], scale_key)
            if self.values.dtype not in ("F8_E4M3", "F8_E4M3FN"):
                raise ValueError("Engram values must be E4M3")
            if self.scales.dtype not in ("F8_E8M0", "F8_E8M0FNU", "U8"):
                raise ValueError("Engram scales must be raw UE8M0")
            if self.values.shape != (self.num_embeddings, self.dim):
                raise ValueError("Engram weight shape mismatch")
            if self.scales.shape != (self.num_embeddings, self.dim // self.block_size):
                raise ValueError("Engram scale shape mismatch")
            # No table-sized Parameters: the EAQ pack deliberately omits these tensors.
            print("PCAI disk Engram:", raw, "TP", get_tensor_model_parallel_rank(),
                  "heads", self.head_start, self.part_n_hash_cols, "O_DIRECT; eager only", flush=True)

        def lookup(self, indices, out, background=False):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("disk Engram cannot execute inside CUDA graphs")
            if out.dtype != torch.bfloat16:
                raise RuntimeError("disk adapter supports only BF16 staging")
            if indices.ndim != 2 or indices.shape[1] != self.n_hash_cols:
                raise ValueError("unexpected hash tensor shape")
            tokens = indices.shape[0]
            if tokens > self.max_tokens:
                raise ValueError("disk staging token budget exceeded")
            expected = (tokens, self.part_n_hash_cols, self.dim)
            if tuple(out.shape) != expected:
                raise ValueError("unexpected output staging shape")
            heads_end = min(self.head_start + self.part_n_hash_cols, self.n_hash_cols)
            ids = indices[:, self.head_start:heads_end].to("cpu").tolist()
            raw = bytearray(tokens * self.part_n_hash_cols * (self.dim + self.dim // 32))
            # Read selected rows only. No readahead, page cache, or row-cache growth.
            values_size = tokens * self.part_n_hash_cols * self.dim
            stride = self.part_n_hash_cols
            valid = []
            for t, token_ids in enumerate(ids):
                for h, row in enumerate(token_ids):
                    if row == -1:  # vLLM DEAD_ID
                        continue
                    if not self.vocab_start_idx <= row < self.vocab_end_idx:
                        raise ValueError("hash row belongs to another TP shard")
                    n = t * stride + h
                    raw[n*self.dim:(n+1)*self.dim] = self.values.read_row(row)
                    begin = values_size + n * (self.dim // 32)
                    raw[begin:begin+self.dim//32] = self.scales.read_row(row)
                    valid.append(n)
            if not tokens:
                return
            packed = torch.frombuffer(raw, dtype=torch.uint8).to(out.device)
            values = packed[:values_size].view(torch.float8_e4m3fn).reshape(
                -1, self.dim).float()
            scales = packed[values_size:].reshape(-1, self.dim // 32).to(torch.int32)
            # Match the upstream UE8M0 decoder: byte -> float32 exponent bits.
            factors = (scales << 23).view(torch.float32).repeat_interleave(32, dim=1)
            decoded = (values * factors).to(torch.bfloat16).reshape(expected)
            # Inactive and padded heads must remain zero, including UE8M0 edge cases.
            mask = torch.zeros(tokens * stride, dtype=torch.bool, device=out.device)
            if valid:
                mask[torch.tensor(valid, device=out.device)] = True
            out.copy_(torch.where(mask.reshape(tokens, stride, 1), decoded, 0))

        def forward(self, indices):
            out = torch.empty((indices.shape[0], self.part_n_hash_cols, self.dim),
                              dtype=torch.bfloat16, device=indices.device)
            self.lookup(indices, out)
            if self.tp_size > 1:
                out = tensor_model_parallel_all_gather(out, dim=1)
            return out[:, :self.n_hash_cols]

        def collapse_huge_pages(self):
            return None

    original_init = impl.Engram.__init__

    @functools.wraps(original_init)
    def init(self, config, quant_config, layout, layer_hash_index,
             use_sequence_parallel, prefix, prefetch_stream=None):
        self._pcai_table_prefix = prefix
        original_init(self, config, quant_config, layout, layer_hash_index,
                      use_sequence_parallel, prefix, prefetch_stream)

    def create(self, layout, hash_index):
        return DiskEmbedding(layout, hash_index, self._pcai_table_prefix)

    impl.Engram.__init__ = init
    impl.Engram._create_embedding = create
    impl.Engram._pcai_disk_installed = True
