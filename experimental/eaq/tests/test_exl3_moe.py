"""Fused EXL3 MoE helpers: physical K tuple and pointer checks; CPU only, the driver is faked."""
from pathlib import Path
import sys
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcai_eaq import exl3_moe
from pcai_eaq.exl3_moe import check_layer, physical_k_tuple


class PhysicalK(unittest.TestCase):
    def test_tuple(self):
        lin = lambda k: N(trellis=torch.empty(2, 2, 16 * k))
        self.assertEqual(physical_k_tuple([{"gate": lin(3), "up": lin(3), "down": lin(3)}] * 4), (3, 3, 3))
        self.assertEqual(physical_k_tuple([{"gate": None, "up": lin(2), "down": lin(4)}]), (2, 2, 4))
        self.assertIsNone(physical_k_tuple([{"gate": lin(3), "up": lin(3), "down": lin(3)},
                                            {"gate": lin(2), "up": lin(2), "down": lin(2)}]))


class PointerChecks(unittest.TestCase):
    def layer(self):
        t = torch.zeros(4)
        lin = N(trellis=t, suh=t, svh=t)
        return N(layer_name="l0"), [{"gate": lin, "up": lin, "down": lin}]

    def test_unavailable_range_is_reported_not_called_overrun(self):
        with patch.object(exl3_moe, "pointer_info", return_value=("host", None, None)):
            _, problems = check_layer(*self.layer())
        self.assertTrue(problems)
        self.assertTrue(all("range unavailable" in p for p in problems))

    def test_overrun_detected(self):
        with patch.object(exl3_moe, "pointer_info", side_effect=lambda ptr: ("host", ptr, 8)):
            _, problems = check_layer(*self.layer())
        self.assertTrue(all("overrun allocation by 8" in p for p in problems))

    def test_unknown_pointer(self):
        with patch.object(exl3_moe, "pointer_info", return_value=(None, None, None)):
            _, problems = check_layer(*self.layer())
        self.assertTrue(all("unknown to CUDA" in p for p in problems))

    def test_driver_errors_on_range_queries(self):
        calls = []

        class Driver:
            def cuPointerGetAttribute(self, out, attr, ptr):
                calls.append(attr)
                if attr == exl3_moe.CU_POINTER_ATTRIBUTE_MEMORY_TYPE:
                    out._obj.value = 1
                    return 0
                return 1  # CUDA_ERROR_INVALID_VALUE

        with patch.object(exl3_moe, "_driver", return_value=Driver()):
            self.assertEqual(exl3_moe.pointer_info(1234), ("host", None, None))


if __name__ == "__main__":
    unittest.main()
