import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.kl import kl_divergence
from torchdiffeq import odeint_adjoint as odeint
from collections import defaultdict

from basic import TemporalPointProcess
from layers.layers import GATLayers, DiffLayers
from layers.nets import (
    PolicyNet, PolicyTransNet, 
    ValueNet, ValueTransNet, 
    IntensityNet, RepresentationNet, PEMValueTransNet
)

from utils.ksubsets import ksubset_sampler,ksubset_logprob
from utils.train_utils import aug_fft_numpy

import layers.diffeq_layers as diffeq_layers

EPS = 1e-9

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def apply_action_dyna(adj : torch.Tensor, control : torch.Tensor, eps=1e-5):
    mask = (adj > eps).to(adj)
    return adj * (1.- control) * mask +  (1.-adj) * (1. -control) * (1.- mask)

def apply_action_pg(adj : torch.Tensor, control : torch.Tensor):
    ## control D*D matrix  1: close,   0: open, 1. - control # of open
    return adj * (1. - control)

def apply_mask(logits : torch.Tensor, mask:  torch.Tensor):
    return mask * (-1e+6) + logits * (1.0 - mask)

def update_mask(mask:torch.Tensor,
                control_memory : torch.Tensor,
                closure_limit : int, controls_limit: int):
    """
    mask : [N, D*(D-1)+num_dummy]
    control_menory: [N, T, D*(D-1)]
    """
    control_memory_sum = torch.sum(control_memory[:,-closure_limit:,:] > 0.3,dim=1)
    assert (control_memory_sum <= closure_limit).all(), \
    f"exceding limits {closure_limit}, get {control_memory_sum[control_memory_sum>closure_limit]}"
    recent_controls = torch.zeros_like(mask)        
    recent_controls[:,:-controls_limit] = control_memory_sum

    return (recent_controls >= closure_limit).to(mask)
        
def aug_perm(hstate):
    bs, D, _ = hstate.shape
    permutation_matrix = torch.zeros(D,D).to(hstate)
    permutation_matrix[torch.arange(0,D),torch.randperm(D)] = 1.0
    permutation_matrix.repeat(bs,1,1)
    
    perm_hstate = permutation_matrix @ hstate
    
    return permutation_matrix, perm_hstate

def aug_fft(hstate, num):
    bs, D, H = hstate.shape
    
    hstate_fft = torch.zeros((bs, num, D, H)).to(hstate)
    hstate_numpy = hstate.clone().detach().cpu().numpy()
    for i in range(bs):
        for j in range(num):
            hstate_fft[i,j,:,:] = torch.Tensor(aug_fft_numpy(hstate_numpy[i,:,:])).to(hstate)
            
    return hstate_fft
            

def bi_contrastive_loss(pem_metric, pos_sim, mage_sim, beta=1.0, temperature=0.1):
    """
    Args:
        pem_metric : [N, 10]
        pos_sim    : [N, 10]
    """
    pos_sim /= temperature
    mage_sim /= temperature 
    term1 = pos_sim[:,0] - mage_sim[:,0]
    pem_sim = torch.exp( -pem_metric / beta )
    # print("pem_sim:",pem_sim)
    ratio = (((1.0 - pem_sim) + EPS) / pem_sim[:,0:1])[:,1:]
    ratio = torch.clip(ratio, min=0.01, max=50)
    # print("ratio:",ratio) 
    # print("pos",pos_sim)
    # print("mage",mage_sim)
    nominator = torch.exp(mage_sim[:,0]) + torch.sum( torch.exp(mage_sim[:,1:]) * ratio, dim=-1)
    # print("nomina:", nominator)
    # print(pos_sim)
    denominator = torch.exp(pos_sim[:,0]) + torch.sum( torch.exp(pos_sim[:,1:]) / ratio, dim=-1)
    
    # print("denominator:", denominator)
    
    
    
    term2 = torch.log(nominator) - torch.log(denominator)
    # print(term1 + term2)
    return term1 + term2
    
    
    
    
class IntensityODEFunc(nn.Module):

    def __init__(
            self,
            represent_fn,
            policy_fn,
            # value_fn,
            state_cfn,
            state_dfn,
            intensity_fn,
        ):
        """
        Args:
            state_cfn: continous state transition function (ODE)
            state_dfn: discreate state transition function (JUMP)
        """
        super().__init__()

        self.state_cfn = state_cfn
        self.state_dfn = state_dfn
        self.intensity_fn = intensity_fn
        self.represent_fn = represent_fn
        self.policy_fn = policy_fn
        # self.value_fn = value_fn
        
    
    def forward(self, t, hstate):
        return self.state_cfn(t, hstate)

    def update_state(self, t, hstate, adj_matrix, obs):
        return self.state_dfn(t, hstate, adj_matrix, obs)
    
    # def get_val(self, hstate, adj):
    #     return self.value_fn(hstate, adj)
    
    def get_intensity(self, hstate):
        return self.intensity_fn(hstate)
    
    def get_representation(self, hstate):
        pos_embed, mage_embed = self.represent_fn(hstate)
        return pos_embed, mage_embed
        
    def get_control_matrix(self, pos_embed, mage_embed, 
                           adj, mask=None,tau=1.):
        """
        Return a binary control matrix to control the graph, 
         * 1 stands for closed , 0 stands for open,
         * all the diagnoal elements are set to 0 
        Args:
            hstate: latent states [N,D,H]
            adj : previous influence adj
        """
        bs = pos_embed.size()[0]
        K, D = self.policy_fn.K, self.policy_fn.D
    
        logits = self.policy_fn(pos_embed, mage_embed)
        logits = apply_mask(logits, mask)
        
        ksample, ksample_ind = ksubset_sampler(logits, 
                                                 K, 
                                              hard=False,
                                              tau=tau)
        
        logprob = ksubset_logprob(logits,ksample_ind, K)
        
        ## sample < or = K elements 
        sample = ksample[:,:-self.policy_fn.K]
        ## get the index of off-diag entries
        b_i, r_i, c_i = torch.where(torch.eye(D).repeat(bs,1,1)==0)
        
        adj_matrix = torch.zeros(bs, D, D).to(logits)
        adj_matrix[b_i,r_i,c_i] = sample.reshape(-1)
        
        return sample, adj_matrix, logprob

class WrapRegularization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, reg, *x):
        ctx.save_for_backward(reg)
        return x

    @staticmethod
    def backward(ctx, *grad_x):
        reg, = ctx.saved_variables
        return (torch.ones_like(reg), *grad_x)

class TimeVariableODE(nn.Module):

    start_time = 0.0
    end_time = 1.0

    def __init__(self, func, atol=1e-6, rtol=1e-6, method="dopri5", energy_regularization=0.01):
        super().__init__()
        self.func = func
        self.atol = atol
        self.rtol = rtol
        self.method = method
        self.energy_regularization = energy_regularization
        self.nfe = 0


    def integrate(self, t0, t1, x0, nlinspace=1, method=None):
        assert nlinspace > 0
        method = method or self.method
        ## steer
        end_time = self.end_time + random.uniform(-0.9,0.9) if self.training else self.end_time
        solution = odeint(
            self,
            (t0, t1, torch.zeros(1).to(x0[0]), x0),
            torch.linspace(self.start_time, self.end_time, nlinspace + 1).to(t0),
            rtol=self.rtol,
            atol=self.atol,
            method=method,
            adjoint_options=dict(norm="seminorm")
        )
        ## TO-DO USE energy momentum
        _, _, energy, xs = solution
        reg = energy * self.energy_regularization
        reg_xs = WrapRegularization.apply(reg, xs)
        
        return reg_xs[0]

    
    def forward(self, s, state):
        """Solves the same dynamics but uses a dummy variable that always integrates [0, 1]."""
        self.nfe += 1
        t0, t1, _, x = state

        ratio = (t1 - t0) / (self.end_time - self.start_time)
        t = (s - self.start_time) * ratio + t0
        with torch.enable_grad():
            x = x.requires_grad_(True)
            dx = self.func(t, x)
            dx = dx * ratio.reshape(-1, *([1] * (dx.ndim - 1))) 

            d_energy = torch.sum(dx * dx) / x.numel()

        if not self.training:
            dx = dx.detach()
        return tuple([torch.zeros_like(t0), torch.zeros_like(t1), d_energy, dx])
    
    
class NueralContJumpProcess(nn.Module):
    
    def __init__(
        self,
        h_dims,
        num_nodes,
        total_time,
        embed_dims = 64,
        max_nodes = 25,
        controls_limit = 8,
        closure_limit = 3,
        hidden_dims=[16],
        cost_matrix = None,
        intervention_coeff = 1.0,
        smooth_coeff = 1.5,
        perminv_coeff = 0.0,
        ode_solver = "dopri5",
        num_policy_layers = 2,
        num_rep_layers = 3,
        contrastive_temperature = 0.5,
        adjs = None,
    ):
        super().__init__()
        self._init_hstate = nn.Parameter(torch.randn(max_nodes, 
                                                     h_dims) / math.sqrt(2*h_dims))
        self.h_dims = h_dims
        self.embed_dims = embed_dims
        self.num_nodes = num_nodes
        self.max_nodes = max_nodes
        self.gamma = 0.99
        self.rho = 0.999
        self.aug_fft_nums = 9
        self.ode_solver_name = ode_solver
        self.constrastive_temperature = contrastive_temperature
        
        state_dfn = GATLayers(c_in=h_dims, c_out=h_dims,
                              num_nodes=num_nodes,
                              hidden_dims=hidden_dims)
        state_cfn = DiffLayers(c_in=h_dims, hidden_dims=hidden_dims, 
                               num_nodes=num_nodes, max_nodes=max_nodes)
        
        represent_func = RepresentationNet(
                                        hdim=h_dims,
                                        embed_dim=embed_dims,
                                        num_nodes=num_nodes, 
                                        num_layers=num_rep_layers
                                    )
        policy_func = PolicyNet(
                        hdim=h_dims, 
                        num_nodes=num_nodes, 
                        num_dummy=controls_limit,
                        num_layers=num_policy_layers
                    )        
        # value_func = ValueNet(hdim=h_dims, num_nodes=num_nodes,num_layers=2)
        
        # policy_func = PolicyTransNet(hdim=h_dims, num_nodes=num_nodes, 
        #                         num_dummy=controls_limit,num_layers=2)        
        # value_func = ValueTransNet(hdim=h_dims, num_nodes=num_nodes,num_layers=2)
        
        intensity_func = IntensityNet(hdim=h_dims, num_nodes=num_nodes,
                                      max_nodes = max_nodes, num_layers=2)
        
        intensity_ode = IntensityODEFunc(
                    represent_fn=represent_func,
                    policy_fn=policy_func,
                    # value_fn=value_func,
                    intensity_fn=intensity_func, 
                    state_cfn=state_cfn, 
                    state_dfn=state_dfn
                )
        
        self.ode_solver = TimeVariableODE(func=intensity_ode)
        self.intervention_cost_matrix = torch.zeros(num_nodes, num_nodes) if cost_matrix is None else cost_matrix
        self.closure_limit = closure_limit
        self.controls_limit = controls_limit
        self.intervention_coeff = intervention_coeff
        self.smooth_coeff = smooth_coeff
        self.perminv_coeff = perminv_coeff
        # self.adjs = nn.Parameter(torch.randn(total_time,num_nodes,num_nodes) / math.sqrt(num_nodes))
        if adjs is None:
            self.adjs = nn.Parameter(torch.randn(max_nodes,max_nodes) / max_nodes )
        else:
            self.adjs = adjs
        # self.prev_state = defaultdict(lambda : torch.zeros(num_nodes,h_dims))
        
        self.pemvalue = PEMValueTransNet(hdim=h_dims, num_nodes=num_nodes, num_layers=2)
        self.target_pemvalue = PEMValueTransNet(hdim=h_dims, num_nodes=num_nodes, num_layers=2)
        
    def _next_hstate(self, t0, t1, hstate, adj_matrix, intensity):
        
        ### discrete jump
        updated_hstate = self.update_state(t0, hstate,
                                           adj_matrix,
                                           intensity
                                           )
        ### continous jump
        next_hstate = self.ode_solver.integrate(t0, t1, 
                                updated_hstate, nlinspace=1, 
                                method=self.ode_solver_name)[-1]
        
        # next_hstate = self.ode_solver.integrate(t0, t1, 
        #                                     updated_hstate, nlinspace=1, 
        #                         method="dopri8" if self.training else "dopri8")[-1]
        
        return next_hstate
    
    def step(self,state, action):
        t0, t1, prev_hstate, intensity, _, _ = state 
        adj_matrix = self.adjs[:self.num_nodes,:self.num_nodes].repeat(1,1,1)
        adj_matrix = apply_action_pg(adj_matrix, control=action)
        next_hstate = self._next_hstate(t0, t1, prev_hstate, adj_matrix, intensity)
        next_intensity = self.get_intensity(next_hstate)

        return t1, next_hstate, next_intensity, adj_matrix
    
    def get_action(self, hstate, prev_adj,mask=None):
        pos_embed, mage_embed = self.get_representation(hstate)
        control_vec, control_mat,logprob = self.get_control_matrix(
            pos_embed, 
            mage_embed,
            adj=prev_adj,
            mask=mask, 
            tau=1e-5
        )

        return control_vec, control_mat, logprob

    def integrate(
            self,
            event_time,
            event_data,
            t0, 
            action_mask=None,
            nlinspace=1,
            policy_learning=False,
            inference = False,
            tau: float = 1.0,
        ):
        """
        Args:
            event_data: (N, T, D)
            event_time: noised time data (N, T)
            t0: (N,) or (1,)
            tau: temperature variable control action sampling
        """
        N, T = event_time.shape
        D = self.num_nodes
     
        init_state = self._init_hstate[:D,:].repeat(N,1,1)  # (D,H) -> (N,D,H)
        state = init_state
    
        intervention_cost_matrix = self.intervention_cost_matrix.repeat(N,1,1).to(event_time)
        mask = torch.zeros(N,D*(D-1)+self.controls_limit).to(event_time) if not action_mask else action_mask.expand(N).to(event_time)
        
        if policy_learning:
            state.require_grad = False
        else:
            state.require_grad = True
        
        t0 = t0 if torch.is_tensor(t0) else torch.Tensor([t0])
        t0 = t0.expand(N).to(event_time)
        # cur_adjs = self.adjs.repeat(N,1,1,1).to(event_time)
        # print("self.adjs",self.adjs.dtype)
        cur_adjs = self.adjs[:D,:D].repeat(N,1,1)
        # print("cur_adjs_0",cur_adjs.dtype)
        self.ode_solver.nfe = 0
        vals = []
        intensities = []

        qz0_mean_list = []
        qz0_logvar_list = []

        control_vec_memory = []
        control_dict = defaultdict(list)
        
        prejump_hidden_states = []

        pre_control_mat = torch.zeros(N,D,D).to(event_time)
        pre_adjs = cur_adjs
        # equally separated time zone
        for i in range(T):
        
            # cur_intensity = self.get_intensity(state)
            # intensities.append(cur_intensity.reshape(N,1,D))
            
            t1_i = event_time[:,i]
            # continuous update at [t0, t1_i)
            # print(i, state)
            
            
            ## use dopri8 and float64 to solve stiff ODE https://github.com/rtqichen/torchdiffeq/issues/58
            # hstate_traj = self.ode_solver.integrate(t0, t1_i, 
            #                                         state, nlinspace=nlinspace, 
            #                                        method="dopri8" if self.training else "dopri8")

            hstate_traj = self.ode_solver.integrate(t0, t1_i, 
                                                    state, nlinspace=nlinspace, 
                                                    method=self.ode_solver_name)

            hiddens = hstate_traj[1]  # (1 + nlinspace, N, D, H)
                      

            hstate = hstate_traj[-1]
            
            prejump_hidden_states.append(hstate)

            ## add sampling
            # qz0_mean, qz0_logvar = hstate[:,:, :self.h_dims], hstate[:, : ,self.h_dims:]
            # epsilon = torch.randn(qz0_mean.size()).to(qz0_mean)
            # z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # qz0_mean_list.append(qz0_mean.reshape(N,1,D, self.h_dims))
            # qz0_logvar_list.append(qz0_logvar.reshape(N,1,D, self.h_dims))


            cur_intensity = self.get_intensity(hstate)
            # print(i, cur_intensity)
            intensities.append(cur_intensity.reshape(N,1,D))
            
            
            if i < T - 1:
                
                if policy_learning:
                    ## generate policy
                    pos_embed, mage_embed = self.get_representation(hstate)
                    control_vec, control_mat,logprob = self.get_control_matrix(pos_embed, 
                                                                               mage_embed,
                                                                               adj=pre_adjs,
                                                                               mask=mask, tau=tau)
                    
                    intervention_cost = torch.sum(control_mat * intervention_cost_matrix,dim=(1,2))
                    # cur_adj = cur_adjs[torch.arange(len(t0)),(t0+0.999).int().cpu().long(),:,:] # (N,T,D,D) -> (n,D,D)
                    # cur_adj = cur_adjs
                    adj_matrix = apply_action_pg(cur_adjs, control_mat)
                    
                    ## difference between two adjacent controls
                    ## we want to smooth out difference
                    # control_diff = torch.sum(torch.abs(control_mat - pre_control_mat), dim=(1,2))
                    control_diff = torch.norm(control_mat - pre_control_mat,dim=(1,2), p=1)
                    
                    #################################################################
                    ### calculate the distance between control_mat and permed out ###
                    perm_matrix, perm_hstate = aug_perm(hstate)
                    control_mat_target, control_mat_target_perm = self.permutation_equivalent_action(
                                                                perm_hstate=perm_hstate,
                                                                perm_matrix=perm_matrix,
                                                                adj=pre_adjs,
                                                                mask=mask, 
                                                                tau=tau
                                                                )
                    #### |pi - p^{T} pi' p| ######
                    perm_distance = torch.norm(control_mat - control_mat_target_perm, p="fro", dim=(1,2))
                    
                    # if i == 0:
                    #     pemval = self.pemvalue(hstate, perm_hstate)
                    #     vals.append(pemval)
                    # elif i == T - 2:
                    #     with torch.no_grad():
                    #         pemval_target = self.target_pemvalue(hstate, perm_hstate)
                    #         vals.append(pemval_target)
                            
                    pemval = self.pemvalue(hstate, perm_hstate)
                    
                    with torch.no_grad():
                        next_hstate = self._next_hstate(t1_i, t1_i+1, hstate, adj_matrix, cur_intensity)
                        
                        ## get next perm state 
                        perm_adj_matrix = apply_action_pg(perm_matrix @ cur_adjs @ perm_matrix.T, control_mat_target)
                        perm_cur_intensity = self.get_intensity(perm_hstate)
                        next_prem_hstate = self._next_hstate(t1_i, t1_i+1,perm_hstate, perm_adj_matrix, perm_cur_intensity)
                        pemval_target = self.target_pemvalue(next_hstate, perm_matrix.T @ next_prem_hstate)
                        
                    pemval_diff = 0.5*(pemval  - (perm_distance + self.gamma * pemval_target))**2
                    
                    ############################ FFT  AUG ##############################
                    fft_hstates = aug_fft(hstate, num=self.aug_fft_nums)
                    pos_embed_fft, mage_embed_fft = self.get_representation(
                                            fft_hstates.reshape(N*self.aug_fft_nums, D, self.h_dims)
                                                    )
                    pos_embed_fft = pos_embed_fft.reshape(N, self.aug_fft_nums, -1)
                    mage_embed_fft = mage_embed_fft.reshape(N, self.aug_fft_nums, -1)
                    
                    pos_embed_perm, mage_embed_perm = self.get_representation(
                                                                perm_hstate
                                                            )
                    
                    cosine_similarity_pos = self.cal_cosine_similarity(pos_embed,pos_embed_perm,pos_embed_fft)
                    cosine_similarity_mage = self.cal_cosine_similarity(mage_embed, mage_embed_perm, mage_embed_fft)
                    
                    pem_metric = self.cal_pem(hstate, perm_hstate, fft_hstates)
                    
                    representation_loss = bi_contrastive_loss(pem_metric, 
                                                              cosine_similarity_pos, 
                                                              cosine_similarity_mage,
                                                              temperature=self.constrastive_temperature)
                    ##################################################################
                

                    control_vec_memory.append(control_vec.clone().detach())
                    control_dict['control_vec'].append(control_vec)
                    control_dict['adj'].append(adj_matrix.reshape(N,1,D,D))
                    control_dict['control'].append(control_mat.reshape(N,1,D,D))
                    control_dict['logprob'].append(logprob.reshape(N,1))
                    control_dict['intervention'].append(intervention_cost.reshape(N,1))
                    control_dict['control_diff'].append(control_diff.reshape(N,1))
                    control_dict['pemval_diff'].append(pemval_diff.reshape(N,1))
                    control_dict['rep_loss'].append(representation_loss.reshape(N,1))
                    control_dict['pos_embed'].append(pos_embed.reshape(N,-1))
                    control_dict['mage_embed'].append(mage_embed.reshape(N,-1))
                    control_dict['pos_embed_perm'].append(pos_embed_perm.reshape(N,-1))
                    control_dict['mage_embed_perm'].append(mage_embed_perm.reshape(N,-1))
                    control_dict['pos_embed_fft'].append(pos_embed_fft[:,0,:].reshape(N,-1))
                    control_dict['mage_embed_fft'].append(mage_embed_fft[:,0,:].reshape(N,-1))
                    # control_dict['perm_loss'].append(perm_loss)
                    # print("cont_mem",(torch.stack(control_vec_memory,dim=1) > 0.3).int())
                    mask = update_mask(mask,torch.stack(control_vec_memory,dim=1),self.closure_limit,self.controls_limit)
                    # print("mask",mask)
                    # generate value function
                    # if i == 0:
                    #     val = self.get_val(hstate,adj_matrix)
                    #     vals.append(val) # (N,) --> [(N,)]
                    
                    pre_control_mat = control_mat
                    pre_adjs = adj_matrix
                    
                else:
                  
                    # print("t0", (t0+0.999).int().cpu().long())
                    ## no intervention
                    # adj_matrix = cur_adjs[torch.arange(len(t0)),(t0+0.999).int().cpu().long(),:,:]  # (N,T,D,D) -> (N,D,D)
                    # print("cur_adj",cur_adjs.dtype)
                    adj_matrix = cur_adjs
                    
                # generate new jump at t1_i
                if policy_learning or inference:
                    # with torch.no_grad():
                    updated_hstate = self.update_state(t1_i, hstate,
                                                           adj_matrix,
                                                           cur_intensity
                                                             )
                else:
                    ## no control fully connected
                    updated_hstate = self.update_state(t1_i, hstate,
                                                           adj_matrix,
                                                           event_data[:,i,:]
                                                           )
                # updated_hstate = hstate 
            t0, state = t1_i, updated_hstate
            
        # if policy_learning:
        #     vals = torch.cat(vals, dim=1)
        intensities = torch.cat(intensities,dim=1)
        # qz0_logvar_tensor = torch.cat(qz0_logvar_list, dim=1)
        # qz0_mean_tensor = torch.cat(qz0_mean_list, dim=1)

        control_dict['hidden_state'] = prejump_hidden_states
        return updated_hstate, intensities, control_dict, vals
    
    
    def reinforce(
        self,
        event_time,
        t0,
        action_mask=None,
        tau:float = 1.0,
    ):
        """
        Update Policy Network by REINFORCE
        """
        #### planning use learnt dynamics
        updated_hstate, intensities, actions, _ = self.integrate(
                                                 event_time,None,
                                                 t0,action_mask,
                                                policy_learning=True,tau=tau)
        
   
        intensities_cost = torch.sum(intensities, dim=-1)[:,1:].clone() # (N,T,D) --> (N,T-1)
        intervention_cost = torch.cat(actions['intervention'],dim=1).clone()  # (N,T-1)
        smooth_cost = torch.cat(actions['control_diff'],dim=1).clone()
        pemval_diff = torch.cat(actions['pemval_diff'],dim=1)
        rep_loss = torch.cat(actions['rep_loss'],dim=1)

        
        N = event_time.shape[0]
        R, T = torch.zeros(N,1).to(event_time), event_time.shape[-1]
        cost2go =  torch.zeros_like(intensities_cost)
        
        for i in reversed(range(T-1)):
            
            R = intensities_cost[:,i:i+1] + R
            cost2go[:,i:i+1] = R
            
#         for i in reversed(range(T-2)):
            
#             pemval_target = perm_distance[:,i] + self.gamma * pemval_target
            

        cost2go = cost2go[:,0]
        # print("intensity",cost2go)
        ### intervention cost
        intervention_cost2go = torch.cumsum(intervention_cost.flip(dims=(1,)),dim=1).flip(dims=(1,))
        intervention_cost2go = intervention_cost2go[:,0]
        # print("interven",intervention_cost2go)
           
        ### action smooth cost
        smooth_cost2go = torch.cumsum(smooth_cost.flip(dims=(1,)),dim=1).flip(dims=(1,))
        smooth_cost2go = smooth_cost2go[:,0]
        # print("smooth",smooth_cost2go)
        
        cost2go = cost2go  + self.intervention_coeff * intervention_cost2go + \
                     self.smooth_coeff * smooth_cost2go 
        
        cost2go = torch.log(cost2go)
        
        pemval_loss = pemval_diff.mean(-1)
        return cost2go.mean(), rep_loss.mean(), pemval_loss.mean(), cost2go
            
            
        
    def neglogprob(
        self,
        event_data,
        event_time,
        t0,
        action_mask=None,
        tau : float = 1.0,
        
    ):
        """
        Update Dynamic Network by MLE
        Args:
            event_data: (N, T, D)
            t0: (N,) or (1,)
        """
        _, intensities, actions, _ = self.integrate(event_time,
                                                    event_data,
                                                    t0,action_mask,
                                                    policy_learning=False,
                                                    tau=tau)
        # qz0_mean, qz0_logvar = qz0
        # log_intensities = torch.log(intensities + 1e-8)
        # x_log_intensities = torch.mul(event_data, log_intensities)
        
        # logp = torch.sum(x_log_intensities,dim=(1,2)) - torch.sum(intensities,dim=(1,2))
        # logp = logp / (torch.sum(event_data,(1,2)) + 1.0)         # additive smoothing
        # return logp.mul(-1.)
        pred = torch.distributions.Poisson(intensities)
        log_likelihood = pred.log_prob(event_data[...,:self.num_nodes]).mean(dim=0).sum()

        # pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(qz0_mean)

        # prior = torch.distributions.Normal(loc=pz0_mean,scale=pz0_logvar)
        # posterior = torch.distributions.Normal(loc=qz0_mean,scale=qz0_logvar)
        # kl = kl_divergence(prior, posterior).mean(dim=0).sum()

        # kl = normal_kl(qz0_mean, qz0_logvar,
        #                         pz0_mean, pz0_logvar).mean(dim=0).sum()

        return -log_likelihood
    
    def inference(
        self,
        t0,
        t1,
        event_time=None,
        action_mask=None,
        tau : float = 1.0,
        policy_learning : bool = False
    ):
        if event_time is None:
            event_time = torch.arange(t0,t1).to(self._init_hstate).unsqueeze(0)

            event_time = event_time + torch.rand(*event_time.shape).to(event_time)
        with torch.no_grad():
            states, intensities, actions, _ = self.integrate(event_time,None,
                                                 t0,action_mask,inference=True,tau=tau,
                                                        policy_learning=policy_learning)
            observations = torch.distributions.Poisson(intensities).sample()
        
        return (observations.squeeze(0),event_time.squeeze(0),states.squeeze(0)), actions, intensities
    
    def forward(
        self,
        event_data,
        event_time,
        t0,
        t1 = None,
        action_mask=None,
        policy_learning = False,
        inference = False,
        tau: float = 1.0,
    ):
        if inference:
            out = self.inference(t0,t1,event_time,action_mask,tau=1e-6, policy_learning = policy_learning)
        elif policy_learning:
            # *out, returns = self.reinforce(event_time,t0,action_mask,tau=tau)
            out = self.reinforce(event_time,t0,action_mask,tau=tau)
        else:
            out = self.neglogprob(event_data,event_time,t0,action_mask, tau=tau)
    
        return out

    # def get_val(self, state, adj):
    #     return self.ode_solver.func.get_val(state, adj)
    
    def get_intensity(self, state):
        return self.ode_solver.func.get_intensity(state)
    
    def get_representation(self, state):
        return self.ode_solver.func.get_representation(state)
        
    def get_control_matrix(self, pos_embed, mage_embed, adj, mask=None,tau=1e-2):
        return self.ode_solver.func.get_control_matrix(
                                        pos_embed, 
                                        mage_embed,
                                        adj,
                                        mask,tau)
    
    def update_state(self, t, state, adj_matrix, obs):
        return self.ode_solver.func.update_state(t, state, adj_matrix,obs)
    
    
    def permutation_equivalent_action(self, perm_hstate, perm_matrix, adj, mask, tau):
        
        diff = 0
        bs, D, H =  perm_hstate.shape
    
        with torch.no_grad():
            
            pos_embed, mage_embed = self.get_representation(perm_hstate)
            adj = perm_matrix @ adj @ perm_matrix.T
            
            _, target_out, _ = self.get_control_matrix(pos_embed, mage_embed, adj, mask, tau)
            target_out_perm = perm_matrix.T @ target_out @ perm_matrix
            
        return target_out, target_out_perm
    
    def cal_cosine_similarity(self, embed, positive_embed, negative_embed):
        bs = embed.shape[0]
        cosine_similarity = torch.zeros((bs, self.aug_fft_nums+1)).to(embed)
        cosine_similarity[:,0] = F.cosine_similarity(embed,positive_embed)
        for i in range(1,self.aug_fft_nums+1):
            cosine_similarity[:,i] = F.cosine_similarity(embed, negative_embed[:,i-1,:])
            
        return cosine_similarity
    
    def cal_pem(self, hstate, perm_hstate, fft_hstate):
        bs = hstate.shape[0]
        pem_metric = torch.zeros((bs, self.aug_fft_nums+1)).to(hstate)
        
        with torch.no_grad():
            pem_metric[:,0] = self.target_pemvalue(hstate, perm_hstate)
            for i in range(1, self.aug_fft_nums+1):
                pem_metric[:,i] = self.target_pemvalue(hstate, fft_hstate[:,i-1,:,:])
        
        return pem_metric
    
    def initialize_target_pemvaluenet(self):
        self.target_pemvalue.load_state_dict(self.pemvalue.state_dict())
        
    def update_target_pemvaluenet(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_pemvalue.state_dict()
        net_state_dict = self.pemvalue.state_dict()
        for key in net_state_dict:
            target_net_state_dict[key] = net_state_dict[key]*(1 - self.rho) + target_net_state_dict[key]* self.rho
        self.target_pemvalue.load_state_dict(target_net_state_dict)

    def update_nodes_nums_adjs_cost(self, new_num, adjs, cost):
        self.update_node_nums(new_num)
        self.update_adj(adjs)
        self.update_intervention_cost(cost)
    
    def update_node_nums(self, new_num):
        """update pruning number"""
        self.num_nodes = new_num
        self.ode_solver.func.policy_fn.D = new_num
        self.ode_solver.func.intensity_fn.D = new_num
        self.ode_solver.func.state_cfn.num_nodes = new_num
        self.ode_solver.func.state_dfn.num_nodes = new_num
        self.pemvalue.D = new_num
        self.target_pemvalue.D = new_num

        for l in self.ode_solver.func.state_cfn.projection.layers:

            if isinstance(l.module,diffeq_layers.basic.NodesBias):
                l.module.D = new_num
    

    def update_adj(self, adjs):
        self.adjs = adjs

    def update_intervention_cost(self, cost):
        self.intervention_cost_matrix = cost    
            
    
    
    
if __name__ == "__main__":
    device = "cpu"
    
    NCJP = NueralContJumpProcess(h_dims=2,num_nodes=4).to(device)
    
    event_data = torch.rand(5,10,4).to(device)
    event_time = torch.tensor(range(10),dtype=torch.float).repeat(5,1).to(device)
    t0 = torch.zeros(5).to(device)
    
    updated_state, intensities_list, actions, vals = NCJP.integrate(event_time,event_data,
                                                         t0,)
    
    logp = NCJP.logprob(event_data,event_time,t0)
    
    pg_loss, returns = NCJP.reinforce(event_time,t0)
    
    (observations,event_time), actions, intensities = NCJP.inference(0,140)
    
    pg_loss = pg_loss.mean()
    pg_loss.backward(retain_graph=True)
    
    loss = logp.mean()
    loss.backward(retain_graph=True)
    
    # for name, p in NCJP.named_parameters():
    #     print(name)
    #     try:
    #         print(p.grad.data)
    #     except AttributeError:
    #         pass