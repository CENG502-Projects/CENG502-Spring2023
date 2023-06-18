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



class Agent(nn.Module):
    def __init__(self,id,num_channels):
        super().__init__()
        self.register_parameter(f'S_{id}',nn.Parameter(torch.ones(num_channels) * 9.6))
        _,S = next(self.named_parameters())
        p = nn.Sigmoid()(S.expand(1,num_channels))
        self.A = torch.bernoulli(p) # Actions
        self.Pi = torch.Tensor(self.A.shape) # policy probability
        self.Pi[self.A==1] = p[self.A==1]
        self.Pi[self.A==0] = 1 - p[self.A==0]
        self.R =  torch.sum(1-self.A,dim=1) # Reward
        
    def forward(self,I):
        N,c,h,w = I.shape
        _,S = next(self.named_parameters())
        p = nn.Sigmoid()(S.expand(N,c))
        self.A = torch.bernoulli(p)
        self.Pi = torch.Tensor(self.A.shape).to(S.device)
        self.Pi[self.A==1] = p[self.A==1]
        self.Pi[self.A==0] = 1 - p[self.A==0]
        self.R = torch.sum(1-self.A,dim=1)
        return I*self.A.view(N,c,1,1).contiguous().expand(N,c,h,w)
    


class DECOR(nn.Module):
    def __init__(self, Model):
        super().__init__()
        self.target_model = Model
        self.agents_list = [] # To keep track of all newly added agents
        self.parse_model()
        
    def parse_model(self):
        """ This method parse the target model and append agents after the activation function of each convolution, the dictionary of the
            target model should be defined for easy parsing """
        n = ''
        modules = torch.nn.Sequential()
        modules_list = [nn.Conv2d]
        for i in self.target_model.state_dict().keys():
            if n != i.split('.')[0]:
                n = i.split('.')[0]
                a = getattr(self.target_model,f'{n}')
                if hasattr(a, '__iter__'):
                    modules = torch.nn.Sequential()
                    for idx in range(0,len(a)):
                        if type(a[idx]) in modules_list and type(a[idx+1]) == nn.ReLU:
                            c = next(a[idx].parameters()).shape[0]
                            modules.add_module(f'{idx}',a[idx])
                            modules.add_module(f'{idx+1}',a[idx+1])
                            agnt = Agent(len(self.agents_list),c)
                            modules.add_module(f'Agent{len(self.agents_list)}',agnt)
                            self.agents_list.append(agnt)
                            idx += 2
                        else:
                            modules.add_module(f'{idx}',a[idx])
                    
                    a = modules
                    setattr(self.target_model,f'{n}',a)
                    
    def forward(self,I):
        return self.target_model(I)
