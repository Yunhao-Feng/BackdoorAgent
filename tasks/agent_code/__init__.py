import json
import os
import time
import openai
from pathlib import Path

from attack import attack_bench
from .ehragent import EhrAgent

def agent_code(args):
    args = attack_bench(args)
    
    save_path = Path(args.save_dir) / Path(args.task) / Path(args.attack)
    os.makedirs(save_path, exist_ok=True)
    
    ehr_agent = EhrAgent(args)
    inference_result = ehr_agent.run()
    
    print(f"\n\nTask {args.task} with {args.attack} attack is finished !!!! \n\n") 