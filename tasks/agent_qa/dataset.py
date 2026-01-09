from torch.utils.data import Dataset
import json
import jsonlines


class StrategyQADataset(Dataset):
    def __init__(self, json_file, split_ratio=0.8, train=True):
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
        except:
            with jsonlines.open(json_file) as reader:
                data = [item for item in reader]

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
            # 'qid': sample['qid'],
            # 'term': sample['term'],
            # 'question': "Question: " + sample['question'],
            'question': sample['question'],
            # 'description': sample['description'] if sample['description'] is not None else "",
            # 'facts': sample['facts'] if sample['facts'] is not None else "",
            # 'decomposition': sample['decomposition'] if sample['decomposition'] is not None else "",
        }
