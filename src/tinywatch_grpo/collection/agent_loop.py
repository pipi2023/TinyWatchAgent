"""Shared TinyWatch agent loop: one tool call per turn, then environment step."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from uuid import uuid4

from tinywatch_grpo.environment.actions import RUNTIME_GUARD_FIELD, action_guard_tool_message
from tinywatch_grpo.environment.env import TinyWatchEnv
from tinywatch_grpo.environment.tools import (
    ENVIRONMENT_VERSION,
    MAX_STEPS_DEFAULT,
    TINYWATCH_TOOL_SCHEMAS,
    REWARD_VERSION,
)


SYSTEM_PROMPT = """你是 TinyWatch 片单 Agent，必须通过工具完成一次约束选片。

规则：
1. 每个 assistant 回合只调用一个工具。不要输出最终片单文本，直到环境结束。
2. 只能打开或写入当前 observation 中出现的 movie_id。
3. 先 search_movies，再 open 或 compare 核验；有导演硬约束时必须 view_crew。
4. 片单部数、总片长、类型、年份、评分满足后再 finalize_watchlist。
5. 充分检索后仍无法满足硬约束时 abort，reason 只能是 no_feasible_watchlist。
6. 不要重复同一动作；被守卫拒绝后根据当前页面改合法动作。
"""

MAX_BLOCKED = 3


class OpenAIChatClient:
    def __init__(
        self,
        model,
        base_url,
        api_key,
        temperature=0.0,
        top_p=1.0,
        timeout=120,
        max_tokens=512,
        thinking=False,
        transport=None,
    ):
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.timeout = timeout
        self.max_tokens = int(max_tokens)
        self.thinking = bool(thinking)
        self.transport = transport
        self.last_usage = {}

    def complete(self, messages, tools):
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": self.max_tokens,
        }
        if self.thinking:
            payload["thinking"] = {"type": "enabled"}
        else:
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p
            if str(self.model).casefold().startswith("deepseek-v4"):
                payload["thinking"] = {"type": "disabled"}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "tinywatch-grpo/0.1",
        }
        url = f"{self.base_url}/chat/completions"
        if self.transport is not None:
            data = self.transport(url, payload, headers, self.timeout)
        else:
            from urllib.request import Request, urlopen

            request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        self.last_usage = data.get("usage") or {}
        message = (data.get("choices") or [{}])[0].get("message") or {}
        return message


def _enforce_serial_tool_call(assistant: dict) -> tuple[dict, list]:
    """Keep the first tool call and drop the rest, matching shopping-grpo-longhorizon."""
    tool_calls = assistant.get("tool_calls") or []
    if len(tool_calls) <= 1:
        return assistant, []
    serial_assistant = dict(assistant)
    serial_assistant["tool_calls"] = [tool_calls[0]]
    return serial_assistant, list(tool_calls[1:])


def parse_tool_call(message: dict) -> tuple[dict | None, str | None]:
    calls = message.get("tool_calls") or []
    if not calls:
        return None, "no_tool_call"
    call = calls[0]
    function = call.get("function") or {}
    name = function.get("name")
    raw = function.get("arguments") or "{}"
    if isinstance(raw, dict):
        arguments = raw
    else:
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            return None, "invalid_tool_arguments"
    if not isinstance(arguments, dict):
        return None, "invalid_tool_arguments"
    return {
        "id": call.get("id") or f"call-{uuid4().hex[:8]}",
        "name": name,
        "arguments": arguments,
        "raw": call,
    }, None


def rollout_task(
    task: dict,
    catalog: dict,
    client,
    *,
    max_steps: int = MAX_STEPS_DEFAULT,
    teacher_model: str | None = None,
) -> dict:
    env = TinyWatchEnv(catalog, task, max_steps=max_steps)
    result = env.reset()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["query"]},
    ]
    steps = []
    blocked = []
    truncations = []
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    error = None
    try:
        while not result["done"]:
            message = client.complete(messages, TINYWATCH_TOOL_SCHEMAS)
            usage = getattr(client, "last_usage", {}) or {}
            for key in usage_total:
                usage_total[key] += int(usage.get(key) or 0)
            message, dropped_tool_calls = _enforce_serial_tool_call(message)
            if dropped_tool_calls:
                kept = (message.get("tool_calls") or [{}])[0]
                truncations.append(
                    {
                        "message_index": len(messages),
                        "kept_tool_call_id": kept.get("id"),
                        "dropped_tool_calls": dropped_tool_calls,
                    }
                )
            assistant = {
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": message.get("tool_calls") or [],
            }
            if message.get("reasoning_content"):
                assistant["reasoning_content"] = message["reasoning_content"]
            messages.append(assistant)
            parsed, parse_error = parse_tool_call(message)
            if parse_error:
                error = parse_error
                break
            env_result = env.step(parsed["name"], parsed["arguments"])
            tool_message = {
                "role": "tool",
                "tool_call_id": parsed["id"],
                "name": parsed["name"],
                "content": env_result["observation"],
            }
            if env_result.get("guard_rejected"):
                tool_message = action_guard_tool_message(
                    {"id": parsed["id"], "function": {"name": parsed["name"]}},
                    env_result.get("guard_reason"),
                    env.last_observation,
                )
                blocked.append({"tool_call": parsed["raw"], "reason": env_result.get("guard_reason")})
                if len(blocked) >= MAX_BLOCKED:
                    error = "too_many_guard_rejections"
                    messages.append(tool_message)
                    steps.append(
                        {
                            "tool_name": parsed["name"],
                            "parameters": parsed["arguments"],
                            "observation": tool_message["content"],
                            "guard_rejected": True,
                            "done": False,
                        }
                    )
                    break
            messages.append(tool_message)
            steps.append(
                {
                    "tool_call": parsed["raw"],
                    "tool_name": parsed["name"],
                    "parameters": parsed["arguments"],
                    "observation": env_result["observation"],
                    "projection": env_result.get("projection"),
                    "reward": env_result["reward"],
                    "done": env_result["done"],
                    "guard_rejected": bool(env_result.get("guard_rejected")),
                }
            )
            result = env_result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback_text = traceback.format_exc()
    else:
        traceback_text = None

    terminal = result.get("terminal_result") or {
        "done": result.get("done"),
        "over": result.get("over"),
        "reward": result.get("reward", 0.0),
        "reward_detail": env.last_reward_detail,
        "watchlist": env.draft,
    }
    return {
        "trajectory_id": str(uuid4()),
        "task_id": task["task_id"],
        "status": "done" if result.get("done") and not error else "error",
        "done": bool(result.get("done")),
        "error": error,
        "traceback": traceback_text,
        "final_reward": terminal.get("reward", 0.0),
        "messages": messages,
        "steps": steps,
        "blocked_tool_calls": blocked,
        "tool_call_truncations": truncations,
        "terminal_result": terminal,
        "token_usage": usage_total,
        "teacher_model": teacher_model or getattr(client, "model", None),
        "thinking": bool(getattr(client, "thinking", False)),
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "environment_version": ENVIRONMENT_VERSION,
        "reward_version": REWARD_VERSION,
        "elapsed_s": None,
    }
