import json
import os
import time
import openai
from pathlib import Path
from .llm import OpenaiEngine
from attack import attack_bench


def load_dataset(dataset_path):
    print(dataset_path)
    with open(dataset_path) as f:
        return json.load(f)

def agent_web(args):
    args = attack_bench(args)
    
    save_path = Path(args.save_dir) / Path(args.task) / Path(args.attack)
    os.makedirs(save_path, exist_ok=True)
    
    web_agent = WebAgent(args)
    inference_result = web_agent.run()
    
    print(f"\n\nTask {args.task} with {args.attack} attack is finished !!!! \n\n") 

class WebAgent:
    def __init__(self, args):
        self.args = args
    
    def run(self):
        args = self.args
        generation_model = OpenaiEngine(
            rate_limit=-1,
            api_key=args.openai.api_key,
            base_url=args.openai.api_url,
            model=args.model,
        )
        
        dataset_dir = os.path.dirname(args.dataset)
        print(f"Dataset directory: {dataset_dir}")
        log_file = f"{args.save_dir}/{args.task}/{args.attack}/web_agent.json"
        print(f"Output file: {log_file}")
        
        dataset = load_dataset(args.dataset)
        print(f"Loaded {len(dataset)} data points")
        