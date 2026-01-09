import json
from pathlib import Path
from ..planning.planning_agent import PlanningAgent

class LanguageAgent:
    def __init__(self, data_path, split, model_name = "glm-4.6", finetune_cot=False, verbose=False) -> None:
        self.data_path = data_path
        self.split = split
        self.split_dict = json.load(open(Path(data_path) / "split.json", "r"))
        self.tokens = self.split_dict[split]
        self.invalid_tokens = []
        self.verbose = verbose
        self.model_name = model_name
        self.finetune_cot = finetune_cot
    
    
    def inference_all(self, data_samples, data_path, save_path, args=None):
        """Inferencing all scenarios"""
        planning_agent = PlanningAgent(verbose=self.verbose)
        inference_result = planning_agent.run_batch(
            data_samples=data_samples,
            data_path=data_path,
            save_path=save_path,
            args=args
        )
        return inference_result