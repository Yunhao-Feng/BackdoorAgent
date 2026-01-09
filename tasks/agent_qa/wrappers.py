import gym
import json
import numpy as np
import os
from pathlib import Path

STRATEGYQA_SPLIT_FILE = {
  "train": "data/agent_qa/database/strategyqa_train.json",
  "dev": "data/agent_qa/database/strategyqa_dev.json",
}


def clean_answer(line):
    if line.strip().lower() == 'no':
        return False
    elif "no" in line.strip().lower() and 'yes' not in line.strip().lower() and 'not' not in line.strip().lower():
        return False
    elif line.strip().lower() == 'yes':
        return True
    elif "yes" in line.strip().lower() and 'no' not in line.strip().lower():
        return True
    else:
        return None

class StrategyQAWrapper(gym.Wrapper):
    def __init__(self, env, split):
        super().__init__(env)
        data_file = STRATEGYQA_SPLIT_FILE[split]
        self.data = json.load(open(data_file, "r"))
        self.data = [(d['question'], d['answer']) for d in self.data]
        self.data_idx = 0
        self.split = split
    
    def reset(self, seed=None, return_info=False, options=None, idx=None):
        self.env.reset(seed=seed, return_info=return_info, options=options)
        try:
            self.env.step('')
        except:
            pass
        self.env.reset(seed=seed, return_info=return_info, options=options)
        self.data_idx = int(np.random.randint(len(self.data))) if idx is None else idx
        observation = f"Question: {self.data[self.data_idx][0]}"
        info = self._get_info()
        return (observation, info) if return_info else observation

    def _get_info(self):
        return {
        "steps": self.steps, 
        "answer": self.answer,
        "question": self.data[self.data_idx][0], 
        "strategy_split": self.split
        }
    
    def get_reward(self, info):
        if info['answer'] is not None:
            gt = self.data[self.data_idx][1]
            pred = clean_answer(info['answer'])
            score = (pred == gt)
            return int(score)
        return 0
    
    def get_metrics(self, info):
        if info['answer'] is not None:
            gt = self.data[self.data_idx][1]
            pred = clean_answer(info['answer'])
            em = (pred == gt)
            # f1 = f1_score(pred, gt)[0]
            return {'reward': em, 'em': em}
        return {'reward': 0, 'em': 0}
    
    def step(self, action):
        # TODO: first step obs does not have question. 
        obs, _, done, info = self.env.step(action)
        reward = self.get_reward(info)
        if done:
            obs = f"Episode finished, reward = {reward}\n"
            info.update({"gt_answer": self.data[self.data_idx][1], "question_idx": self.data_idx})
            info.update(self.get_metrics(info))
        return obs, reward, done, info
    
    def __len__(self):
        return len(self.data)
        

class LoggingWrapper(gym.Wrapper):
    def __init__(self, env, folder="trajs", file_id=None):
        super().__init__(env)
        self.trajs = []
        self.traj = {"observations": [], "actions": []}
        base_dir = getattr(getattr(env.env, "args", None), "save_dir", "result")
        task_name = getattr(getattr(env.env, "args", None), "task", "agent_qa")
        attack_name = getattr(env.env, "attack", getattr(env, "attack", ""))
        self.folder = str(Path(base_dir) / task_name / attack_name / folder)
        os.makedirs(self.folder, exist_ok=True)
        self.file_id = np.random.randint(0, 10000000) if file_id is None else file_id
        self.file_path = f"{self.folder}/{self.file_id}.json"
    
    def __len__(self):
        return len(self.env.data)
    
    def reset(self, seed=None, return_info=False, options=None, idx=None):
        output = self.env.reset(seed=seed, return_info=return_info, options=options, idx=idx)
        observation = output[0] if return_info else output
        self.traj = {"observations": [observation], "actions": []}
        return output
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.traj["observations"].append(obs)
        self.traj["actions"].append(action)
        if done:
            self.traj.update(info)
        return obs, reward, done, info
    
    def update_record(self):
        if len(self.traj) > 0:
            self.trajs.append(self.traj)
            self.traj = {"observations": [], "actions": []}
  
    def write(self):
        self.update_record()
        with open(self.file_path, "w") as f:
            json.dump(self.trajs, f)
            print(f"Saved trajs to trajs/{self.file_id}.json")
        
    def close(self):
        self.write()
