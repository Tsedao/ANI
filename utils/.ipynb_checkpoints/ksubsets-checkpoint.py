import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
EPSILON = np.finfo(np.float32).tiny


def ksubset_sampler(logits, k, hard=True, tau=1.0):
    m = torch.distributions.gumbel.Gumbel(
            torch.zeros_like(logits), 
            torch.ones_like(logits),
            )
    g = m.sample()
    logits = logits + g
    
    # continuous top k
    khot = torch.zeros_like(logits)
    khot_list = []
    onehot_approx = torch.zeros_like(logits)
    for i in range(k):
        khot_mask = torch.max(1.0 - onehot_approx, 
                              torch.tensor([EPSILON]).to(logits))
        
        logits = logits + torch.log(khot_mask)
        onehot_approx = F.softmax(logits / tau, dim=-1)
        # print("one_hot:",onehot_approx,torch.argmax(onehot_approx,dim=-1))
        khot_list.append(torch.argmax(onehot_approx,dim=-1))
        khot = khot + onehot_approx

    if hard:
        # straight through
        khot_hard = torch.zeros_like(khot)
        val, ind = torch.topk(khot, k, dim=-1)
        khot_hard = khot_hard.scatter_(-1, ind, 1)
        res = khot_hard - khot.detach() + khot
    else:
        res = khot

    return res, khot_list


def batch_ksubset_sampler(logits, ks, hard=True, tau=0.1):
    N = logits.shape[0]
    
    res_list = []
    batch_khot_list = []
    for b_i in range(N):
        res, khot_list = ksubset_sampler(logits[b_i],int(ks[b_i]),
                                             hard,tau)
        res_list.append(res.unsqueeze(0))
        batch_khot_list.append(khot_list)
        
    return torch.cat(res_list,dim=0), batch_khot_list


# def ksubset_logprob(logits, khot_list, k):
#     """Order matters"""
#     index = khot_list   # index order of which val is sampled the first [0,2,3]
#     khot_mask = torch.ones_like(logits)
#     logprob = torch.zeros(1).to(logits)
#     for i in range(k):
        
#         logits = logits + torch.log(khot_mask)
#         logprob_i =  F.log_softmax(logits,dim=-1)
#         logprob = logprob + logprob_i[index[i]]
        
#         khot_mask[index[i]] = 0.
    
#     return logprob


def ksubset_logprob(logits, khot_list, k):
    """Order matters"""
    N = logits.shape[0]
    index = khot_list   # index order of which val is sampled the first [0,2,3]
    khot_mask = torch.ones_like(logits)
    logprob = torch.zeros(N,1).to(logits)
    for i in range(k):
        
        logits = logits + torch.log(khot_mask)
        logprob_i =  F.log_softmax(logits,dim=-1)
        logprob = logprob + torch.gather(logprob_i,dim=1,index=index[i].unsqueeze(1))
        
        khot_mask.scatter_(index=index[i].unsqueeze(1), dim=1, value=1e-10)
    
    return logprob 


def batch_ksubset_logprob(logits, batch_khot_list, ks):
    
    N = logits.shape[0]
    logprobs = []
    for i in range(N):
        logprob = ksubset_logprob(logits[i],batch_khot_list[i],int(ks[i]))
        logprobs.append(logprob)
    
    return torch.cat(logprobs,dim=0)


def k_gumbel_sampler(logits, hard=True, tau=0.1):
    m = torch.distributions.gumbel.Gumbel(
            torch.zeros_like(logits), 
            torch.ones_like(logits),
            )
    g = m.sample()
    logits = logits + g
    
    ohot_approx = F.softmax(logits / tau, dim=-1)
    ind = torch.argmax(ohot_approx, dim=-1)
    if hard:
        # straight through
        ohot_hard = torch.zeros_like(logits)
        ohot_hard[:,ind] = 1.
        res = ohot_hard 
    else:
        res = ohot_approx
        
    return res, ind
    
def k_gumbel_logprob(logits, ind):
    """
    Args:
        ind : [N,1]
    """
    logsoft = F.log_softmax(logits,dim=-1)
    return torch.gather(logsoft,1,ind)