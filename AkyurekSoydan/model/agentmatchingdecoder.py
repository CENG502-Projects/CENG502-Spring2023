"""
    This file is created under the CENG 502 Project.

    version: 23-06-14-03-01
    
    Classes:
    1. AgentMatchingDecoder
    2. FeedForward
        
"""


import math
import copy
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

class AgentMatchingDecoder(nn.Module):
    # c: hidden dimensions (see c in the paper notation)
    # feat_res: Resolution of feature space (i.e. h and w, assuming h = w)
    # im_res: Resolution of original image
    def __init__(self, device, heads, c, feat_res, dropout = 0.1):
        super().__init__()
        
        self.device = device
        self.c = c
        self.d_k = c // heads
        self.d_k_sqrt = math.sqrt(self.d_k)

        self.h = heads
        
        self.qa_linear = nn.Linear(c, c)
        self.ks_linear = nn.Linear(c, c)

        self.qq_linear = nn.Linear(c, c)
        self.ka_linear = nn.Linear(c, c)
        
        self.vs_linear = nn.Linear(c, c)
        
        self.ffn = FeedForward(c)             

       
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(c, c)

        self.feat_res = feat_res

    
    def forward(self,
                tok_agent,              # agent tokens, F_a_head
                enc_feat_supp,          # encoded support feauters, F_s_head
                enc_feat_query,         # encoded query feauters, F_q_head
                mask=None
                ):      
        bs = tok_agent.size(0)
        
        hw = enc_feat_supp.shape[1]

        qa = self.qa_linear(tok_agent)#.view(bs, -1, self.h, self.d_k)
        #print("qa.shape = ", qa.shape)     
        ks = self.ks_linear(enc_feat_supp)#.view(bs, -1, self.h, self.d_k)
        #print("ks.shape = ", ks.shape)

        qq = self.qa_linear(enc_feat_query)#.view(bs, -1, self.h, self.d_k)
        ka = self.ka_linear(tok_agent)#.view(bs, -1, self.h, self.d_k)

        vs = self.vs_linear(enc_feat_supp)#.view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * c
        #qa = qa.transpose(1,2)     
        #print("qa.shape = ", qa.shape)        
        #ks = ks.transpose(1,2)             
        #print("ks.shape = ", ks.shape)
        
        #qq = qq.transpose(1,2)             
        #ka = ka.transpose(1,2)             

        #vs = vs.transpose(1,2)             

        # calculate scores
        scores_as = torch.matmul(qa, ks.transpose(-2, -1)) /  self.d_k_sqrt
        scores_qa = torch.matmul(qq, ka.transpose(-2, -1)) /  self.d_k_sqrt
        
        #scores_as = scores_as.transpose(1, 2).contiguous().view(bs, hw, -1)
        #scores_qa = scores_qa.transpose(1, 2).contiguous().view(bs, -1, hw)
        #scores_as = torch.reshape(scores_as, (bs, hw, -1))
        #scores_qa = torch.reshape(scores_qa, (bs, -1, hw))

        ####
        #print("scores_as.shape = ", scores_as.shape)
        #print("scores_qa.shape = ", scores_qa.shape)

        
        # TODO: can we implement it without for loops? (to make it faster)
        # Aligning Matrix
        align_mat =  torch.empty(bs,hw,hw)
        for i in range(hw):
            for j in range(hw):
                align_mat[:,i,j] = (torch.argmax(scores_as[:,:,i], dim=-1) == torch.argmax(scores_qa[:,j,:], dim=-1))

        align_mat = (align_mat.to(self.device) - 1) * 1e6
        #print("align_mat.shape = ", align_mat.shape)
        #print("align_mat = ", align_mat)

        scores_sa = scores_as.transpose(-1,-2)
        scores_aq = scores_qa.transpose(-1,-2)

        scores_qs = F.softmax((torch.matmul(scores_sa, scores_aq) + align_mat), dim=-1)

        #print("scores_qs.shape = ", scores_qs.shape)
        #print("vs.shape = ", vs.shape)

        dec_feat_query = self.ffn(torch.matmul(scores_qs.transpose(1, 2), vs))
        dec_feat_query = dec_feat_query.contiguous().view(bs, self.c, self.feat_res, -1) 
        #print("dec_feat_query.shape = ", dec_feat_query.shape)

        return dec_feat_query


class FeedForward(nn.Module):
    def __init__(self, c, d_ff=2048, dropout = 0.1):
        super().__init__() 

        self.linear_1 = nn.Linear(c, 3*c) #d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(3*c, c) #d_ff, c)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
