from torch.utils.data import Dataset
import json

class AgentDriverDataset(Dataset):
    def __init__(self, json_file, split_ratio=0.8, train=True):
        with open(json_file, 'r') as file:
            data = json.load(file)
        split_index = int(len(data) * split_ratio)
        if train:
            self.data = data[:split_index]
        else:
            self.data = data[split_index:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")
        sample = self.data[idx]
        # for key in sample:
            # if sample[key] is None:
            #     print(f"None found in key: {key}, index: {idx}")
            #     input()
        # Convert data to the required format, process it if necessary
        return {
            'token': sample['token'],
            'ego': sample['ego'],
            'perception': sample['perception'],
            'commonsense': sample['commonsense'] if sample['commonsense'] is not None else "",
            'experiences': sample['experiences'] if sample['experiences'] is not None else "",
            'chain_of_thoughts': sample['chain_of_thoughts'] if sample['chain_of_thoughts'] is not None else "",
            'reasoning': sample['reasoning'] if sample['reasoning'] is not None else "",
            'planning_target': sample['planning_target'] if sample['planning_target'] is not None else "",
        }