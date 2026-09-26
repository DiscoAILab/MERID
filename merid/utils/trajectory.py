"""Run trajectory: system prompts, logged agent decisions with their reasoning, and token and cost totals over all LLM calls."""
from __future__ import annotations

import json
import os


class Trajectory:
    def __init__(self, model="", config=None, stream_path=None):
        self.info = {
            "model_stats": {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0},
            "config": {"model": model, "system_prompts": {}, "space": (config or {}).get("space", [])},
        }
        if config:
            self.info["config"].update({k: v for k, v in config.items() if k != "space"})
        self.trajectory = []
        self._step = 0
        self._stream = open(stream_path, "a", encoding="utf-8") if stream_path else None

    def add_system_prompt(self, name, text):
        self.info["config"]["system_prompts"][name] = text

    def record_call(self, in_tokens=0, out_tokens=0, cost=0.0):
        s = self.info["model_stats"]
        s["api_calls"] += 1
        s["input_tokens"] += int(in_tokens)
        s["output_tokens"] += int(out_tokens)
        s["cost"] = round(s["cost"] + float(cost), 6)

    def log(self, type, reasoning="", prompt="", choice=None, observation=None, **fields):
        self._step += 1
        step = {"step": self._step, "type": type, "prompt": prompt, "reasoning": reasoning,
                "choice": choice, "observation": observation, **fields}
        self.trajectory.append(step)
        if self._stream:
            self._stream.write(json.dumps(step, ensure_ascii=False) + "\n"); self._stream.flush()
        return step

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"info": self.info, "trajectory": self.trajectory}, f, ensure_ascii=False, indent=2)

    def close(self, path=None):
        if path:
            self.save(path)
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
