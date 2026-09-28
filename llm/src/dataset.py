import numpy as np
import torch
from torch.utils.data import Dataset

class MemmapTokenDataset(Dataset):
    """
    Memmap 기반 token dataset.
    tokens.dat를 메모리에 올리지 않고 필요한 부분만 slice로 제공.
    """
    def __init__(self, memmap_file, max_length, stride):
        self.tokens = np.memmap(memmap_file, dtype=np.int32, mode='r')
        self.max_length = max_length
        self.stride = stride
        self.num_samples = max(0, (len(self.tokens) - max_length) // stride)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length
        input_ids = torch.tensor(self.tokens[start:end], dtype=torch.long)
        target_ids = torch.tensor(self.tokens[start+1:end+1], dtype=torch.long)
        return input_ids, target_ids


def collate_fn(batch):
    """
    Batch collate function for DataLoader.
    Stacks input_ids and target_ids into tensors.
    """
    input_batch = torch.stack([x for x, y in batch])
    target_batch = torch.stack([y for x, y in batch])
    return input_batch, target_batch