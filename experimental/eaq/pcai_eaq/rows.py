"""Bounded direct-I/O safetensors row reader, with no full-table mmap/cache."""
import json
import mmap
import os
import struct
from pathlib import Path

ALIGN = 4096
MAX_HEADER = 16 * 1024 * 1024


def head_partition(head_sizes, rank, world):
    if not head_sizes or any(n <= 0 for n in head_sizes):
        raise ValueError("invalid head sizes")
    if world < 1 or not 0 <= rank < world:
        raise ValueError("invalid TP rank/world")
    width = (len(head_sizes) + world - 1) // world
    if (world - 1) * width >= len(head_sizes):
        raise ValueError("TP leaves a rank with no hash heads")
    start = rank * width
    return start, width, sum(head_sizes[:start]), sum(head_sizes[:start + width])


def safe_path(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("tensor file outside staging directory or absent")
    return path


class DirectTensor:
    """One reusable aligned page window. Unsupported O_DIRECT fails closed."""
    def __init__(self, root, filename, key):
        self.path = safe_path(root, filename)
        with self.path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError("truncated safetensors header")
            size = struct.unpack("<Q", raw)[0]
            if not 2 <= size <= MAX_HEADER:
                raise ValueError("invalid safetensors header size")
            header = json.loads(f.read(size))
        if key not in header:
            raise ValueError(f"tensor {key} not in {filename}")
        meta = header[key]
        if meta["dtype"] not in ("F8_E4M3", "F8_E4M3FN", "F8_E8M0", "F8_E8M0FNU", "U8"):
            raise ValueError("expected one-byte native FP8 or scale tensor")
        self.dtype = meta["dtype"]
        self.shape = tuple(meta["shape"])
        if len(self.shape) != 2 or min(self.shape) <= 0:
            raise ValueError("expected positive 2D table")
        begin, end = meta["data_offsets"]
        self.start = 8 + size + begin
        self.row_bytes = self.shape[1]
        self.end = 8 + size + end
        if begin < 0 or end - begin != self.shape[0] * self.row_bytes:
            raise ValueError("invalid tensor data extent")
        if self.end > self.path.stat().st_size or self.row_bytes > ALIGN:
            raise ValueError("truncated tensor or oversized row")
        self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
        self.buffer = mmap.mmap(-1, ALIGN * 2)
        # Exercise filesystem/alignment support before model loading begins.
        try:
            self.read_row(0)
        except BaseException:
            self.close()
            raise

    def read_row(self, row):
        if not 0 <= row < self.shape[0]:
            raise IndexError("Engram row out of range")
        offset = self.start + row * self.row_bytes
        aligned = offset // ALIGN * ALIGN
        inner = offset - aligned
        length = ((inner + self.row_bytes + ALIGN - 1) // ALIGN) * ALIGN
        view = memoryview(self.buffer)[:length]
        try:
            count = os.preadv(self.fd, [view], aligned)
            if count < inner + self.row_bytes:
                raise OSError("short direct-I/O row read")
            return bytes(view[inner:inner + self.row_bytes])
        finally:
            view.release()

    def close(self):
        if getattr(self, "fd", None) is not None:
            os.close(self.fd)
            self.fd = None
        if getattr(self, "buffer", None) is not None:
            self.buffer.close()
            self.buffer = None
