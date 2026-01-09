import json
import os
import time
from pathlib import Path

from attack import attack_bench
from .language.language_agent import LanguageAgent

def agent_driver(args):
    # args = attack_bench(args)
    language_agent = LanguageAgent(
        data_path=args.data_path,
        split=args.split,
        model_name=args.model,
        finetune_cot=args.finetune_cot,
        verbose=args.verbose,
    )
    
    save_path = Path(args.save_path) / Path(args.task) / Path(args.attack)
    os.makedirs(save_path, exist_ok=True)
    with open(args.test_samples_dir, "r") as f:
        test_data_samples = json.load(f)[0:200]
        
    inference_result = language_agent.inference_all(
        data_samples=test_data_samples, 
        data_path=Path(args.data_path) / Path(args.split), 
        save_path=save_path,
        args=args
    )
    
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/result.json", "w", encoding="utf-8") as f:
        json.dump(inference_result, f, ensure_ascii=False, indent=4)
    
    print(f"\n\nTask {args.task} with {args.attack} attack is finished !!!! \n\n") 