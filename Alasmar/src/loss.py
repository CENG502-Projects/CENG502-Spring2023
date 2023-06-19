import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToTensor,ToPILImage
from torchvision.io import read_image
from torchvision import models
from PIL import Image
import os
import io
import pandas as pd
import math
from torch.optim import Adam

class CustomLoss(nn.Module):
    def __init__(self, penalty):
        super().__init__()
        self.penalty = penalty # How much to penalize incorrect drop

    def forward(self, agents_list, y_predicted, target):
        loss = 0.0
        N,_ = y_predicted.size()
        y_predicted = y_predicted.argmax(axis = 1)
        t = torch.Tensor(y_predicted.shape).to(y_predicted.device)
        t[y_predicted==target] = 1
        t[y_predicted!=target] = self.penalty
        for a in agents_list:
            R = a.R*t
            loss += torch.mean(torch.sum(torch.log(a.Pi),dim=1)* R)
        return loss # The aim is to maximize rewards