# BackdoorAgent: A Unified Framework for Backdoor Attacks on LLM-based Agents

<div align="center">

**A research-focused benchmark for studying backdoor behaviors in agentic LLM systems**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#requirements)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#license)

</div>

---

## Overview

BackdoorBench is a modular codebase for evaluating backdoor behaviors across multiple agentic tasks (e.g., QA, web navigation, autonomous driving planning, code/medical agents). It provides:

- **Unified task runner** with YAML + CLI configuration.
- **Multiple backdoor attack implementations** (e.g., agentpoison, trojanrag, demonagent, badchain).
- **Task-specific pipelines** with structured logging and result artifacts.
- **Reproducible experiment setup** via config-driven overrides.

---

## Repository Structure

```

.
├── attack/                 # Attack implementations
├── configs/                # Default + task-specific configs
├── runs/                   # Entry points and run scripts
├── tasks/                  # Task-specific pipelines (agent_qa, agent_web, agent_driver, agent_code)
├── llm_client.py           # Unified LLM client wrapper
├── utils.py                # Utilities (merging configs, printing, etc.)
└── result/                 # Outputs (created at runtime)

````

---

## Requirements

- **Python 3.9+**
- Core dependencies typically include:
  - `openai`
  - `torch`
  - `transformers`
  - `tqdm`
  - `tenacity`

---

## Quick Start

### 1) Configure API access

Edit `configs/default.yaml` with your API key and endpoint:

```yaml
openai:
  api_key: "<YOUR_KEY>"
  api_url: "<YOUR_ENDPOINT>"
````

### 2) Run a task

```bash
python runs/run.py --task agent_qa --attack normal --model qwen3-max
```

### 3) Explore outputs

Results and logs are written under:

```
result/<task>/<attack>/
```

---

## Tasks

| Task           | Description                        | Entry Module         |
| -------------- | ---------------------------------- | -------------------- |
| `agent_qa`     | StrategyQA-style QA with retrieval | `tasks/agent_qa`     |
| `agent_web`    | Web navigation agent               | `tasks/agent_web`    |
| `agent_driver` | Autonomous driving planning        | `tasks/agent_driver` |
| `agent_code`   | Code/medical coding agent          | `tasks/agent_code`   |

---

## Attacks

Attack methods are configured in `configs/task_configs/<task>.yaml`. Examples include:

* `agentpoison`
* `trojanrag`
* `demonagent`
* `badagent`
* `badchain`
* `advagent`

Each attack exposes tunable parameters such as trigger sequences, poisoned ratios, and target keywords.

---

## Configuration System

Configuration is composed from:

1. `configs/default.yaml`
2. `configs/task_configs/<task>.yaml`
3. CLI overrides (e.g., `--task`, `--attack`, `--model`)

Configs are merged at runtime by `runs/run.py`.

---

## Example Experiments

Run a batch of attacks for `agent_code`:

```bash
bash run.sh
```

Run individual attacks:

```bash
python runs/run.py --task agent_driver --attack poisonedrag --model qwen3-max
python runs/run.py --task agent_qa --attack badchain --model qwen3-max
```

---

## Reproducibility Notes

* Seed handling and dataset splits are task-specific.
* If you introduce new models, update `runs/run.py` and task configs as needed.
* Large runs can be parallelized, but ensure output paths do not collide.

---

## Citation

If you use this repository in academic work, please cite the corresponding paper (if applicable):

```bibtex
@misc{backdoorbench,
  title        = {BackdoorBench: A Modular Benchmark for Backdoor Evaluation in Agentic LLMs},
  author       = {Yunhao Feng, Yige Li, Yutao Wu, Yingshui Tan, Yanming Guo, Yifan Ding, Kun Zhai, Xingjun Ma, Yugang Jiang},
  year         = {2025},
  howpublished = {\url{https://arxiv.org/abs/2601.04566}}
}
```

---

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file in the repository root for details.

> Note: Apache-2.0 permits commercial use, modification, and distribution, provided you follow the license terms (e.g., preserving copyright notices).

---

## Acknowledgements

We thank the community for open-source tooling that enables reproducible research in LLM safety and evaluation.