# This file is the part of CENG 502 - Advanced Deep Learning, METU, Course Project
# We implement attention mechanisms as given in the equations of the AAFormer paper.

# Assumptions
# --------------------------------------------------------------------------
# 1-) Authors do not specify if the attention mechanism of Agent Learning Decoder 
# is multi-head or not, so we assume it is not multi-head.
# For Representation Encoder and Agent Matching Decoder, they mention the 
# equations are implemented with multi-head mechanism.
# We implement the attention as if it is multi-head and setting num_heads=1
# will make it a single head.
#
# 2-) Authors do not mention about normalization or dropouts that are used in
# transformer encoder and decoders. We assume the model has transformer-like 
# architecture and we use normalization and dropouts as paper mentions
# "to integrate adaptive prototypes as agent into affinitybased
# FSS via a transformer encoder-decoder architecture"

import ot

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# See supplementary material:
# "The hidden dimension of MLP layer is set to three times of input hidden dimension"
class FeedForward(nn.Module):
    def __init__(self, c, dropout = 0.1):
        super().__init__() 

        self.linear_1 = nn.Linear(c, 3*c)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(3*c, c)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


# This attention mechanism is not the same with original attention.
# Here we implement eqn.3 and eqn.4, i.e. 
# S = softmax(QK^T / sqrt(d_k) + M)
# There will be another algorithm, Optimal Transport (eqn.5-6) before
# we scale the attention with Value.
class Attention_Eqn3(nn.Module):
    def __init__(self, hidden_dims, hw):
        super().__init__()
        
        self.d_k = hidden_dims
        self.d_k_sqrt = 1 #math.sqrt(self.d_k)
        self.W_a_Q = nn.Linear(hidden_dims, hw) #c,h*w
        self.W_s_K = nn.Linear(hidden_dims, hw) #c,h*w
    
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, F_a, F_s, M):
        
        # F_a has shape (batchsize, num_tokens, c=d_k)
        # F_s has shape (batchsize, h*w, c=d_k)
        Q_a = self.W_a_Q(F_a)  # Get Query, shape (batchsize, numtokens, c=d_k)
        K_s = self.W_s_K(F_s)  # Get Key, shape (batchsize, hw, c=d_k)
        
        # Transposed Key has shape (batchsize, c=d_k, hw)
        # Such that the result QK has shape (batchsize, numtokens, hw)
        # This corresponds to K x hw dimensions of M in eqn.4, see page 8 first sentence
        QK = torch.matmul(Q_a, K_s.transpose(1,2)) / self.d_k_sqrt
        S = self.softmax(QK + M)
        
        return S

class Attention_Eqn7(nn.Module):
    def __init__(self, hidden_dims, num_tokens):
        super().__init__()

        self.W_s_V = nn.Linear(hidden_dims, hidden_dims)
        
        #self.ffn = nn.Linear(hidden_dims, hidden_dims)
        self.ffn = FeedForward(hidden_dims)

    def forward(self, F_s, S_hat):

        V_s = self.W_s_V(F_s)

        # V_s has shape (batchisze, hw, c)
        # S_hat has shape (batchsize, num_tokens, hw)
        return self.ffn(torch.matmul(S_hat, V_s)) # Alternative to Eqn.7, w.r.t Fig.4(a)'s flow
        # return torch.matmul(self.ffn(S_hat), V_s) # Eqn.7, cannot be multiplied.


# This is where we implement Agent Learning Decoder's equations
class AgentLearningDecoderAttention(nn.Module):
    
    def __init__(self, cuda,  c, hw, num_layers, num_tokens, num_heads=1, sinkhorn_reg = 1e-1):
        super().__init__()
        
        self.cuda = cuda
        self.device = "cuda" if cuda else "cpu"

        self.reg = sinkhorn_reg
        self.d_k = c #// num_heads # Assumption: this decoder has single-head attention, (nothing is mentioned in the paper)
        self.num_tokens = num_tokens

        self.attn_eqn3 = Attention_Eqn3(self.d_k, hw)
        self.attn_eqn7 = Attention_Eqn7(self.d_k, self.num_tokens)
        
    
    def forward(self, F_a, F_s, M_s, bypass_ot = False, max_iter_ot = 1000):
        
        F_a = F_a.to(self.device)

        # Step 1: Masked Cross Attention between (F_a, F_s_hat)
        # This part is the implementation of eqn.3 and eqn.4
        # Return Part Mask S (see Fig.3 (a))
        # --------------------------------------------------------
        
        M_s = M_s.unsqueeze(1)
        # Flatten M_s  from shape (batchsize, 1, h, w)
        # to shape (batchsize, 1, h*w)
        M_s = torch.flatten(M_s, start_dim=2)
        
        # See page 8, first sentence, N is the duplication of M
        # for each token of the agent tokens.
        # N has shape (batchsize, numtokens, hw)
        N = M_s.repeat(1,self.num_tokens,1) 
                 
        M = torch.where(N == 1, 0, float('-inf'))
        # Debug: Check M has zeros in it
        # print ((M == 0).nonzero(as_tuple=True)[0])
        
        # Get the "masked attention weight matrix"
        # S has shape (batchsize, numtokens, hw)
        S = self.attn_eqn3(F_a, F_s, M)
        
        # Step 2: Get refined masks with Optimal Transport Algorithm
        # This part is the implementation of eqn.5 and eqn.6
        # --------------------------------------------------------
        lambd = self.reg # Regularization term for OT
        batchsize = len(S)

        # Question: can we flatten the batch dimension and compute OT? 
        # Assumption: but that would correlate the batches and OT would jointly
        # solve the problem. Instead, we want optimal transport within
        # a single image?
        # Note: Optimal Transport library we use, is defined for 1D or 2D distributions
        # We need to compute it for every image in the batch separately
        # See example https://pythonot.github.io/auto_examples/plot_OT_2D_samples.html
        # Note: Optimal transport matrix elements may be faulty, we added a bypass 
        # option in case it causes a problem during training.
        S_hat = S

        if not bypass_ot:
            S_hat = torch.zeros_like(S)
            for i in range(batchsize):
                
                # Debug: Check if mask has any ones in it
                # print(torch.any(M_s == 1))
                # M_s has shape (batchsize, 1, h*w) (as flattened before)
                # Get a single mask with shape (h*w):
                feat_mask = M_s[i].squeeze(0)  

                # Step 2.1: Get the foreground pixels of S (refer to Fig.4)
                S_fg = S[i][:,feat_mask == 1]
                
                # Step 2.2: Compute the Optimal Transport of Sinkhorn algorithm
                # The algorithm is taken from: https://github.com/rythei/PyTorchOT/tree/master
                # Assumption: Paper doesn't explain the "similarity matrix", we assume it is
                # S we obtain from attention (as the notation matches), and S^fg is the foreground
                # pixels of S, obtained by the support feature mask == 1 entries.
                cost_mat = torch.ones_like(S_fg) - S_fg 
                
                # num_fg_pix is the number of foreground pixels, i.e. N in the eqn.5-6
                num_fg_pix = S_fg.shape[1]  # Every sample in a batch has different num of foreground pixels

                # Assumption: If the feature mask has no 1, we skip this image.
                if(num_fg_pix == 0):
                    print(">> WARNING: Sample with no feature mask occurred.")
                    continue

                # See the definition at https://pythonot.github.io/all.html#ot.sinkhorn
                # and eqn.6 given in the paper, a = 1/K and b = 1/N
                # a and b are multiplied by a matrix of ones as paper mentions
                # "1 denotes the vector of all ones in the appropriate dimension" for eqn.6
                # We assumed T*1 is the multiplication of [KxN][Nx1] --> [Kx1] 
                # T.T * 1 is [NxK][Kx1] --> [Nx1] in eqn.6. 
                # Hence, a has dimension K, b has dimension N. 
                
                #with torch.no_grad():
                a = (1/self.num_tokens) * torch.ones(self.num_tokens).to(self.device)
                b = (1/num_fg_pix) * torch.ones(num_fg_pix).to(self.device)
                T_single = ot.sinkhorn(a, b, cost_mat, lambd, numItermax=max_iter_ot) # has shape (numtokens, num_fg_pix)
                
                # Debug: check the number of foreground pixels for every sample in a batch
                # Output: they are all different.
                # print(num_fg_pix)
                # Debug: Check if optimal transport matrix makes sense
                # Output: Looks like every element is the same, but they differ for every image
                # print(T_single[0])
                

                # Step 2.3: Zero-pad the background pixels to revert the refined mask's 
                # dimension to same dimension as S (refer to Fig.4)
                # Equivalently, place the optimal transport output to foreground pixels' location
                T_single_expanded = torch.zeros_like(S[i])
                T_single_expanded[:, feat_mask == 1] = T_single

                S_hat[i] = T_single_expanded

        # Assumption: In fig.4, there is a weighted pooling module before obtaining the final tokens
        # We assume that the pooling scheme is the multiplication of V^s part in eqn.7 
        # (and eqn.12 for figure4(b)'s weighted pooling)
        # However, it seems like there is a mismatch between eqn.7 and fig.4 (a)
        # In eqn.7  the result of OT, S_hat, is directly passed through a
        # Feed-Forward Network (FFN) and then scaled by Values (Weighted Pooling).
        # When we look at fig.4 (a), it is vice versa.
        # In eqn.12, the equation matches with fig.4 (b), and the weighted pooling part
        # of fig.4 (b) is the same with fig.4 (a). Therefore, we assume eqn.7 is meant to be
        # F_hat_a = FFN(S_hat * V_s), i.e. first pooling then FFN is applied to refined masks.
        # This choice is also supported by the dimensions of matrices we need to multiply, 
        # and FFN is supposed to be the final projection of agent tokens.
        
        # Assumption: Definition of V_s is not provided by the paper. From the notations used
        # by the paper, we assume V_s = F^s * W^V_s where W^V_s is a learnable parameter.
        # Step 3-4: Weighted Pooling and FFN
        # --------------------------------------------------------
        F_a_hat = self.attn_eqn7(F_s, S_hat) 
        
        return F_a_hat
    

class AgentLearningDecoder(nn.Module):
    def __init__(self, cuda, c, hw, num_layers, num_tokens, num_heads=1, sinkhorn_reg = 1e-1):
        super().__init__()
        self.cross_attn = AgentLearningDecoderAttention(cuda, c, hw, num_layers, num_tokens, num_heads, sinkhorn_reg)

    def forward(self, F_a, F_s, M_s, bypass_ot = False, max_iter_ot=10):
        
        b, h, w = M_s.shape
        F_a_hat = self.cross_attn(F_a, F_s, M_s, bypass_ot, max_iter_ot)

        # TODO/Assumption: There is no specification about normalization or dropout 
        # If we follow the notations used in the paper, this file should contain everything 
        # paper provides. The flow of original decoder is different from the AAFormer's decoder figures.
        return F_a_hat