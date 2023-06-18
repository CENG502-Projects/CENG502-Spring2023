import torch
import pandas as pd
from torch.utils.data import Dataset

class UserBusinessDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df['user_id'].values)
        self.positives = torch.LongTensor(df['business_id'].values)
        self.negatives = torch.LongTensor(df['negative_id'].values)
        self.weights =  torch.LongTensor(df['weight'].values)

    def __getitem__(self, index):
        return self.users[index], self.positives[index], self.negatives[index], self.weights[index]

    def __len__(self):
        return self.users.size(0)