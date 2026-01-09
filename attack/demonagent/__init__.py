from dataclasses import dataclass
from pathlib import Path


@dataclass
class DemonAgentConfig:
    poisoned_percents: float
    target_keyword: str
    trigger_sequence: str
    injection_count: int
    fragment_count: int
    overlap_size: int
    anchor_start: str
    anchor_end: str
    temporal_seed: str
    encoding_alphabet: str
    telemetry_prefix: str
    fragment_header: str
    embedding: str
    retriever_model_path: str


class DemonAgent:
    """Placeholder attacker that ensures DemonAgent assets are initialised."""

    def __init__(self, args) -> None:
        self.args = args

    def _build_config(self) -> DemonAgentConfig:
        cfg = getattr(self.args, "demonagent", None)
        if cfg is None:
            raise ValueError("Missing demonagent configuration in args")

        return DemonAgentConfig(
            poisoned_percents=getattr(cfg, "poisoned_percents", 0.0),
            target_keyword=getattr(cfg, "target_keyword", ""),
            trigger_sequence=getattr(cfg, "trigger_sequence", ""),
            injection_count=getattr(cfg, "injection_count", 0),
            fragment_count=getattr(cfg, "fragment_count", 3),
            overlap_size=getattr(cfg, "overlap_size", 0),
            anchor_start=getattr(cfg, "anchor_start", "<dg>"),
            anchor_end=getattr(cfg, "anchor_end", "</dg>"),
            temporal_seed=getattr(cfg, "temporal_seed", "demon-fsm"),
            encoding_alphabet=getattr(cfg, "encoding_alphabet", ""),
            telemetry_prefix=getattr(cfg, "telemetry_prefix", "telemetry"),
            fragment_header=getattr(cfg, "fragment_header", ""),
            embedding=getattr(cfg, "embedding", "dpr-ctx_encoder-single-nq-base"),
            retriever_model_path=getattr(cfg, "retriever_model_path", "data/model_cache/dpr-ctx_encoder-single-nq-base"),
        )

    def run(self):
        config = self._build_config()
        cache_dir = Path(getattr(self.args, "memory_cache_dir", "data/agent_driver/memory"))
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Persist the resolved configuration for reproducibility.
        setattr(self.args, "demonagent", config)
        return self.args