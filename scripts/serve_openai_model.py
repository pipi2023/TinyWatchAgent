#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat completions server for TinyWatch eval/GRPO rollouts."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor

from tinywatch_grpo.collection.local_client import TransformersToolClient


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="tinywatch-agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def load_model(model_path: str, dtype_name: str):
    load_kwargs = {"trust_remote_code": True}
    config = AutoConfig.from_pretrained(model_path, **load_kwargs)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        **load_kwargs,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, processor


def main():
    args = parse_args()
    model, processor = load_model(args.model, args.dtype)
    app = FastAPI(title="TinyWatch OpenAI-compatible serve")
    clients: dict[tuple[float, float], TransformersToolClient] = {}

    def get_client(temperature: float, top_p: float) -> TransformersToolClient:
        key = (float(temperature), float(top_p))
        if key not in clients:
            clients[key] = TransformersToolClient(
                model,
                processor,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=args.max_new_tokens,
                enable_thinking=False,
            )
        return clients[key]

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": args.served_model_name, "object": "model", "owned_by": "tinywatch"}],
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "model": args.served_model_name}

    @app.post("/v1/chat/completions")
    def chat_completions(payload: dict[str, Any]):
        messages = payload.get("messages") or []
        tools = payload.get("tools") or []
        temperature = float(payload.get("temperature", 0.0))
        top_p = float(payload.get("top_p", 1.0))
        client = get_client(temperature=temperature, top_p=top_p)
        if payload.get("max_tokens"):
            client.max_new_tokens = int(payload["max_tokens"])
        message = client.complete(messages, tools)
        return JSONResponse(
            {
                "id": "chatcmpl-tinywatch",
                "object": "chat.completion",
                "model": args.served_model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if message.get("tool_calls") else "stop",
                    }
                ],
                "usage": client.last_usage,
            }
        )

    print(
        json.dumps(
            {
                "serving": args.model,
                "served_model_name": args.served_model_name,
                "url": f"http://{args.host}:{args.port}/v1",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
