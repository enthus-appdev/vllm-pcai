import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcai_eaq.rows import DirectTensor, head_partition, safe_path


def fixture(path):
    values = bytes(range(256)) * 100
    meta = json.dumps({"weight": {"dtype": "F8_E4M3", "shape": [100, 256],
                                   "data_offsets": [0, len(values)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(meta)) + meta + values)
    return values


class RowsTest(unittest.TestCase):
    def test_tp_heads_match_upstream_partition(self):
        self.assertEqual(head_partition((11, 13, 17, 19), 0, 2), (0, 2, 0, 24))
        self.assertEqual(head_partition((11, 13, 17, 19), 1, 2), (2, 2, 24, 60))

    def test_padded_last_rank(self):
        self.assertEqual(head_partition((11, 13, 17), 1, 2), (2, 2, 24, 41))

    def test_invalid_partition(self):
        for args in [((), 0, 1), ((1,), 1, 1), ((1,), 0, 2), ((0, 1), 0, 1)]:
            with self.assertRaises(ValueError):
                head_partition(*args)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                safe_path(root, "../outside")

    def test_real_direct_io(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "table.safetensors"
            expected = fixture(p)
            reader = DirectTensor(root, p.name, "weight")
            try:
                for row in [0, 15, 16, 31, 98, 99, 0]:
                    self.assertEqual(reader.read_row(row), expected[row*256:(row+1)*256])
                for row in [-1, 100]:
                    with self.assertRaises(IndexError):
                        reader.read_row(row)
                self.assertEqual(len(reader.buffer), 8192)
            finally:
                reader.close()
                reader.close()

    def test_truncation(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "table.safetensors"
            fixture(p)
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                DirectTensor(root, p.name, "weight")

    def test_reject_short_header(self):
        with tempfile.TemporaryDirectory() as root:
            p = Path(root) / "table.safetensors"
            p.write_bytes(b"short")
            with self.assertRaises(ValueError):
                DirectTensor(root, p.name, "weight")


if __name__ == "__main__":
    unittest.main()
