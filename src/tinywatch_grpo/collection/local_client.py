"""Local Transformers chat client that returns OpenAI-style tool calls."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from uuid import uuid4

import torch


TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
FUNCTION_CALL_RE = re.compile(
    r"<function=(?P<name>[^\s>]+)>\s*(?P<body>.*?)</function>",
    re.DOTALL,
)
ARG_RE = re.compile(
    r"<parameter=(?P<key>[^\s>]+)>\s*(?P<value>.*?)\s*</parameter>",
    re.DOTALL,
)


def parse_generated_tool_calls(text: str) -> list[dict]:
    """Parse Qwen-style tool call markup into OpenAI tool_calls objects."""
    calls = []
    for match in TOOL_CALL_RE.finditer(text or ""):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        arguments = payload.get("arguments", {})
        if not name:
            continue
        if isinstance(arguments, str):
            raw_arguments = arguments
        else:
            raw_arguments = json.dumps(arguments, ensure_ascii=False)
        calls.append(
            {
                "id": f"call-{uuid4().hex[:8]}",
                "type": "function",
                "function": {"name": name, "arguments": raw_arguments},
            }
        )
    if calls:
        return calls
    for match in FUNCTION_CALL_RE.finditer(text or ""):
        name = match.group("name")
        arguments = {
            arg.group("key"): arg.group("value")
            for arg in ARG_RE.finditer(match.group("body") or "")
        }
        for key, value in list(arguments.items()):
            try:
                arguments[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
        calls.append(
            {
                "id": f"call-{uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return calls


def messages_for_chat_template(messages: list[dict]) -> list[dict] | None:
    """Qwen chat templates call ``arguments|items``, so arguments must be dicts."""
    normalized = deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, dict):
                continue
            if not isinstance(arguments, str):
                return None
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            function["arguments"] = parsed
    return normalized


class TransformersToolClient:
    """Generate one assistant turn with optional tools using a local HF model."""

    def __init__(
        self,
        model,
        processor,
        *,
        temperature: float = 0.8,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", None) or processor
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = bool(enable_thinking)
        self.last_usage = {}
        self.device = next(model.parameters()).device

    def complete(self, messages, tools):
        rendered = messages_for_chat_template(messages)
        if rendered is None:
            raise ValueError("messages contain non-object tool call arguments")
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "tools": tools,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        try:
            prompt = self.processor.apply_chat_template(rendered, **template_kwargs)
        except TypeError:
            template_kwargs.pop("chat_template_kwargs", None)
            prompt = self.processor.apply_chat_template(rendered, **template_kwargs)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_len = int(encoded["input_ids"].shape[-1])
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["top_p"] = self.top_p
        with torch.inference_mode():
            output = self.model.generate(**encoded, **generate_kwargs)
        completion_ids = output[0, prompt_len:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
        self.last_usage = {
            "prompt_tokens": prompt_len,
            "completion_tokens": int(completion_ids.shape[-1]),
            "total_tokens": prompt_len + int(completion_ids.shape[-1]),
        }
        tool_calls = parse_generated_tool_calls(text)
        if not tool_calls and self.temperature > 0:
            generate_kwargs["temperature"] = min(1.3, self.temperature + 0.2)
            with torch.inference_mode():
                output = self.model.generate(**encoded, **generate_kwargs)
            completion_ids = output[0, prompt_len:]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=False)
            self.last_usage["completion_tokens"] = int(completion_ids.shape[-1])
            self.last_usage["total_tokens"] = prompt_len + int(completion_ids.shape[-1])
            tool_calls = parse_generated_tool_calls(text)
        content = TOOL_CALL_RE.sub("", text)
        content = FUNCTION_CALL_RE.sub("", content).strip()
        for token in (
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "</think>",
            "<think>",
            "<tool_call>",
            "</tool_call>",
        ):
            content = content.replace(token, "")
        content = content.strip()
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message
