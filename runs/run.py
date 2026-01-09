import sys
sys.path.append(".")

import argparse
import copy
import importlib
import os
from pathlib import Path
import sys
import yaml
import torch


MODEL_CHOICES = [
    "glm-4.6",
    "kimi-k2",
    "llama3",
    "gpt-52-1211-global",
    "qwen3-max",
    "gemini-3-pro-preview",
    "DeepSeek-R1-671B",
    "claude_sonnet4_5",
    "qwen3-max-preview",
]



from utils import deep_merge, dict_to_namespace, pretty_print_ns


def parse_args():
    parser = argparse.ArgumentParser(description="Run LanguageAgent with YAML + CLI args")
    parser.add_argument("--attack", type=str, default="normal", help="Type of attack method")
    parser.add_argument("--task", type=str, default="agent_code", help="Task name")
    parser.add_argument("--model", type=str, default=MODEL_CHOICES[0], choices=MODEL_CHOICES, help="Model name for unified API")
    cfg = parser.parse_args()

    default_cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8")) or {}
    task_cfg_path = Path(f"configs/task_configs/{cfg.task}.yaml")
    if not task_cfg_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_cfg_path}")
    task_cfg = yaml.safe_load(task_cfg_path.read_text(encoding="utf-8")) or {}

    file_cfg = deep_merge(default_cfg, task_cfg)
    merged_cfg = dict_to_namespace(deep_merge(file_cfg, vars(cfg)))
    # Backward compatibility: ensure all components see the unified model name
    merged_cfg.model_name = merged_cfg.model
    model_root = Path(merged_cfg.save_dir) / merged_cfg.model
    merged_cfg.save_dir = str(model_root)
    if hasattr(merged_cfg, "save_path"):
        merged_cfg.save_path = str(Path(merged_cfg.save_path) / merged_cfg.model)
    if hasattr(merged_cfg, "logs_path"):
        merged_cfg.logs_path = str(model_root / merged_cfg.task / "logs")
    if hasattr(merged_cfg, "llm"):
        merged_cfg.llm = merged_cfg.model
    if hasattr(merged_cfg, "advagent"):
        merged_cfg.advagent.prompter_model = merged_cfg.model
    if hasattr(merged_cfg, "badagent"):
        merged_cfg.badagent.update_model_name = False
    return merged_cfg


def main():
    args = parse_args()

    if getattr(args, "device", "cpu") == "cuda":
        if torch is None or not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU for this run.")
            args.device = "cpu"

    os.environ["OPENAI_API_KEY"] = args.openai.api_key
    os.environ["OPENAI_BASE_URL"] = args.openai.api_url

    pretty_print_ns(args)

    task_modules = {
        "agent_driver": ("tasks.agent_driver", "agent_driver"),
        "agent_web": ("tasks.agent_web", "agent_web"),
        "agent_code": ("tasks.agent_code", "agent_code"),
        # "agent_research": ("tasks.agent_research", "agent_research"),
        "agent_qa": ("tasks.agent_qa", "agent_qa"),
    }

    if args.task not in task_modules:
        raise ValueError(f"Unsupported task: {args.task}")

    module_name, func_name = task_modules[args.task]
    task_module = importlib.import_module(module_name)
    task_runner = getattr(task_module, func_name)
    task_runner(args)


if __name__ == "__main__":
    main()
