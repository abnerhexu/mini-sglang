from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

_NAME_PATTERN = re.compile(r'"name"\s*:\s*"([^"\\]*)')
_ARGUMENTS_PATTERN = re.compile(r'"arguments"\s*:\s*\{(.*)', re.DOTALL)
_KEY_PATTERN = re.compile(r'(?:^|,)\s*"([^"\\]+)"\s*:')


@dataclass(frozen=True)
class ToolSchema:
    name: str
    required_keys: tuple[str, ...] = ()
    sandbox_type: str = "python"
    preparation_level: str = "runtime"


@dataclass(frozen=True)
class ToolCallEvidence:
    text: str
    name_prefix: str | None
    exact_name: str | None
    candidate_names: tuple[str, ...]
    observed_keys: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, object]


class StreamingToolCallParser:
    """Incrementally extracts evidence from JSON-style function calls.

    The parser intentionally accepts incomplete JSON. It is used only for
    speculation; complete calls must still be parsed and validated by the
    execution path before a tool is run.
    """

    def __init__(self, max_buffer_chars: int = 64 * 1024) -> None:
        self.max_buffer_chars = max_buffer_chars
        self._buffer = ""

    @property
    def buffer(self) -> str:
        return self._buffer

    def reset(self) -> None:
        self._buffer = ""

    def feed(self, delta: str, tools: Iterable[ToolSchema]) -> ToolCallEvidence:
        self._buffer = (self._buffer + delta)[-self.max_buffer_chars :]
        schemas = tuple(tools)
        names = tuple(tool.name for tool in schemas)

        name_match = _NAME_PATTERN.search(self._buffer)
        name_prefix = name_match.group(1) if name_match else None
        exact_name = name_prefix if name_prefix in names else None
        candidates = (
            tuple(name for name in names if name.startswith(name_prefix))
            if name_prefix is not None
            else names
        )

        observed_keys: tuple[str, ...] = ()
        arguments_match = _ARGUMENTS_PATTERN.search(self._buffer)
        if arguments_match:
            observed_keys = tuple(dict.fromkeys(_KEY_PATTERN.findall(arguments_match.group(1))))

        complete = False
        try:
            parsed = json.loads(self._buffer)
            complete = _contains_complete_call(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        return ToolCallEvidence(
            text=self._buffer,
            name_prefix=name_prefix,
            exact_name=exact_name,
            candidate_names=candidates,
            observed_keys=observed_keys,
            complete=complete,
        )


def parse_complete_tool_call(text: str) -> ParsedToolCall | None:
    """Find the first complete JSON tool call in otherwise free-form output."""

    decoder = json.JSONDecoder()
    for offset, character in enumerate(text):
        if character not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            continue
        call = _find_complete_call(value)
        if call is not None:
            return call
    return None


def _find_complete_call(value: object) -> ParsedToolCall | None:
    if isinstance(value, dict):
        name = value.get("name")
        arguments = value.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            return ParsedToolCall(name=name, arguments=arguments)
        for child in value.values():
            call = _find_complete_call(child)
            if call is not None:
                return call
    elif isinstance(value, list):
        for child in value:
            call = _find_complete_call(child)
            if call is not None:
                return call
    return None


def _contains_complete_call(value: object) -> bool:
    return _find_complete_call(value) is not None
