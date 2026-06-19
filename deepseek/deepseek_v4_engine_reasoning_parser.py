# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.parser.deepseek_v4 import DeepSeekV4Parser
from vllm.parser.engine.adapters import make_adapters

(
    DeepSeekV4ParserReasoningAdapter,
    DeepSeekV4ParserToolAdapter,
) = make_adapters(DeepSeekV4Parser)

__all__ = [
    "DeepSeekV4ParserReasoningAdapter",
    "DeepSeekV4ParserToolAdapter",
]
