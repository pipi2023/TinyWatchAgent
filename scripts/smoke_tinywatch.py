#!/usr/bin/env python3
"""CPU smoke without models or Flash API."""

from tinywatch_grpo.smoke import run_cpu_smoke
import json

if __name__ == "__main__":
    result = run_cpu_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
