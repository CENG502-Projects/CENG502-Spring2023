"""
    version: 23-06-13-21-00
    
    Disclaimer: The following Transformer code is adapted from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

    Classes:
    1. RepresentationEncoder
    2. PositionalEncoder
    3. EncoderLayer
    4. MultiHeadAttention
    5. Norm
    6. FeedForward

    Methods:
    1. attention
        
"""



import math
import copy
import torch
from torch import nn
import torch.nn.functional as F


class RepresentationEncoder(nn.Module):
    def __init__(self, c, hw, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(c,hw)
        self.layers = nn.ModuleList([copy.deepcopy(EncoderLayer(c, heads)) for i in range(N)])       
        self.norm = Norm(c)

    def forward(self, feat, mask = None):
        #b_f, c_f, h_f, w_f = feat.shape
        b_f, h_f, w_f, c_f = feat.shape
        hw_f = h_f * w_f

        feat = feat.contiguous().view(b_f, c_f, -1).permute(0,2,1)

        x = self.pe(feat)

        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class PositionalEncoder(nn.Module):
    def __init__(self, c, hw):
        super().__init__()
        
        pe = torch.zeros(hw, c)
        position = torch.arange(0, hw, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, c, 2).float() * -(math.log(10000.0) / c))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe
    

class EncoderLayer(nn.Module):
    def __init__(self, c, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(c)
        self.norm_2 = Norm(c)
        self.attn = MultiHeadAttention(heads, c)
        self.ff = FeedForward(c)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, c, dropout = 0.1):
        super().__init__()
        
        self.c = c
        self.d_k = c // heads
        self.h = heads
        
        self.q_linear = nn.Linear(c, c)
        self.v_linear = nn.Linear(c, c)
        self.k_linear = nn.Linear(c, c)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(c, c)
    
    def forward(self, q, k, v, mask=None):      
        bs = q.size(0)
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * c
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.c)
        
        output = self.out(concat)
    
        return output


class Norm(nn.Module):
    def __init__(self, c, eps = 1e-6):
        super().__init__()
    
        self.size = c
        
        # two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=(-2,-1), keepdim=True)) / (x.std(dim=(-2,-1), keepdim=True) + self.eps) + self.bias
        return norm


class FeedForward(nn.Module):
    def __init__(self, c, d_ff=2048, dropout = 0.1):
        super().__init__() 

        self.linear_1 = nn.Linear(c, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, c)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) 
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output
