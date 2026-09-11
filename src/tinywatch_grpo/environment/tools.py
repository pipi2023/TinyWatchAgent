"""TinyWatch tool schema v1: the only actions a model may call.

JSON Schema and Python dispatch stay in this file so Flash collection, SFT,
GRPO and evaluation cannot drift apart.
"""

from __future__ import annotations


ENVIRONMENT_VERSION = "tinywatch-environment-v1"
OBSERVATION_VERSION = "tinywatch-observation-v1"
TOOL_SCHEMA_VERSION = "tinywatch-tool-schema-v1"
REWARD_VERSION = "tinywatch-reward-v1"

MAX_STEPS_DEFAULT = 14
SEARCH_TOP_K = 8
COMPARE_MAX_ITEMS = 4
OBS_CHAR_BUDGET = 1800

TOOL_NAMES = (
    "search_movies",
    "open",
    "view_crew",
    "compare",
    "add_to_watchlist",
    "view_draft",
    "finalize_watchlist",
    "abort",
)


def _schema(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TINYWATCH_TOOL_SCHEMAS = [
    _schema(
        "search_movies",
        "按关键词检索影片。query 应短，包含片名、类型、导演或年代线索，不要复制整段用户需求。",
        {
            "query": {"type": "string"},
            "genre": {"type": "string"},
            "year_min": {"type": "integer"},
            "year_max": {"type": "integer"},
            "min_rating": {"type": "number"},
        },
        ["query"],
    ),
    _schema(
        "open",
        "打开最新 observation 中出现的影片详情（片长、票数、原名）。movie_id 必须原样取自当前页面。",
        {"movie_id": {"type": "string"}},
        ["movie_id"],
    ),
    _schema(
        "view_crew",
        "查看已出现在最新 observation 中的影片导演名单。有导演硬约束时必须核验后再写入片单。",
        {"movie_id": {"type": "string"}},
        ["movie_id"],
    ),
    _schema(
        "compare",
        "并排比较 2–4 部已出现在最新 observation 中的影片。",
        {
            "movie_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 4,
            }
        },
        ["movie_ids"],
    ),
    _schema(
        "add_to_watchlist",
        "把已打开、比较或查看过演职人员的影片写入片单。超出目标部数时覆盖最旧的一部。",
        {"movie_id": {"type": "string"}},
        ["movie_id"],
    ),
    _schema(
        "view_draft",
        "查看当前片单：已选影片、总片长、评分与类型覆盖。无参数，必须传 {}。",
    ),
    _schema(
        "finalize_watchlist",
        "不可撤销的终止动作。仅当片单部数正确、总片长不超、类型/年份/评分/导演硬约束已满足，且关键影片已 open 或 compare 核验后调用。无参数，必须传 {}。",
    ),
    _schema(
        "abort",
        "主动结束且不提交片单。仅当充分检索后仍无法同时满足类型、片长预算和硬约束时调用。",
        {
            "reason": {
                "type": "string",
                "enum": ["no_feasible_watchlist"],
            }
        },
        ["reason"],
    ),
]


def tool_call_to_action(name, parameters):
    """Normalize a tool call into a stable action record for the environment."""
    parameters = parameters or {}
    if name not in TOOL_NAMES:
        raise KeyError(f"unknown tool: {name}")
    return {"name": name, "parameters": dict(parameters)}
