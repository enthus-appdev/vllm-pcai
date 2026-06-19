# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 reasoning parser on the streaming parser engine.

DeepSeek V4 / R1 reasoning format::

    <think>reasoning...</think>answer...

The chat template pre-seeds the opening ``<think>`` for thinking turns, so
the model output usually starts *inside* reasoning and only emits the closing
``</think>``. The engine therefore starts in ``REASONING`` (when thinking is
on) and treats a stray opening ``<think>`` as a no-op self-loop.

Reasoning only: DeepSeek tool calls use the tokenizer's ``<｜tool▁calls▁begin｜>``
special-token format, which the dedicated ``deepseek_v4`` *tool* parser already
handles. This engine config carries no tool transitions and is paired with that
tool parser at the serving layer, exactly the side-by-side mode the engine
adapters were built for. The win here is the engine's chunk-size-invariant
reasoning extraction: large multi-token deltas from MTP / speculative decoding
no longer leak ``</think>`` or raw reasoning into content the way the legacy
single-token-delta parser does.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    Transition,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

THINK_START = "<think>"
THINK_END = "</think>"

# Strings that mean "thinking off" when passed as chat_template_kwargs.thinking
# (DeepSeek exposes an enum: adaptive/enabled/disabled — not a bare bool).
_THINKING_OFF = frozenset({"false", "disabled", "off", "no", "none", "0"})


def _thinking_on(chat_template_kwargs: dict) -> bool:
    """DeepSeek-V4-Flash is a thinking model and we default it on; honor an
    explicit ``thinking``/``enable_thinking`` (bool or enum string) override."""
    for key in ("thinking", "enable_thinking"):
        val = chat_template_kwargs.get(key)
        if val is None:
            continue
        if isinstance(val, str):
            return val.strip().lower() not in _THINKING_OFF
        return bool(val)
    return True


@functools.cache
def deepseek_v4_config(thinking: bool = True) -> ParserEngineConfig:
    return ParserEngineConfig(
        name="deepseek_v4",
        initial_state=ParserState.REASONING if thinking else ParserState.CONTENT,
        terminals={
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        },
        token_id_terminals={
            "THINK_START": THINK_START,
            "THINK_END": THINK_END,
        },
        transitions={
            # Pre-seeded or re-emitted <think> while already reasoning: no-op.
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # Absorb a duplicate </think> after reasoning already ended.
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        # DeepSeek answers can legitimately open with whitespace/newlines.
        strip_trailing_reasoning_whitespace=False,
        tool_args_json=False,
    )


class DeepSeekV4Parser(ParserEngine):
    """DeepSeek V4 ``<think>``/``</think>`` reasoning on the parser engine.

    Tool calls are left to the ``deepseek_v4`` tool parser (different token
    format); this engine extracts reasoning only.
    """

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = _thinking_on(chat_kwargs)
        kwargs.setdefault(
            "parser_engine_config",
            deepseek_v4_config(thinking=self.thinking_enabled),
        )
        super().__init__(tokenizer, tools, **kwargs)

    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            return None, model_output
        return super().extract_reasoning(model_output, request)
