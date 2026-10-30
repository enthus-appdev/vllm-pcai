"""Compatibility guards for the patched vLLM/vllm-exl3 internals; CPU only."""
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pcai_eaq.compat import ALLOW_UNTESTED, check_vllm, require


class Target:
    def method(self, a, b=None):
        return a


class Compat(unittest.TestCase):
    def vllm(self, version):
        return patch.dict(sys.modules, {"vllm": types.SimpleNamespace(__version__=version)})

    def test_qualified_version(self):
        with self.vllm("0.32.0"):
            check_vllm()
        with self.vllm("0.32.1.dev12+gabc"):
            check_vllm()

    def test_other_version_fails_unless_allowed(self):
        with self.vllm("0.33.0"), patch.dict(os.environ, {ALLOW_UNTESTED: ""}):
            with self.assertRaisesRegex(RuntimeError, "qualified on 0.32"):
                check_vllm()
        with self.vllm("0.33.0"), patch.dict(os.environ, {ALLOW_UNTESTED: "1"}):
            check_vllm()

    def test_require(self):
        require(Target, "method", ("self", "a", "b"))
        with self.assertRaisesRegex(RuntimeError, "no longer exists"):
            require(Target, "missing")
        with self.assertRaisesRegex(RuntimeError, r"now takes \['self', 'a', 'b'\]"):
            require(Target, "method", ("self", "a"))


if __name__ == "__main__":
    unittest.main()
