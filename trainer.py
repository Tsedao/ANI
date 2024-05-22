import os
import math
import time
import yaml
import json
import itertools
import torch.nn as nn
import random
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

from utils.scheduler import CyclicCosineDecayLR
import matplotlib.pyplot as plt
import logging_utils

from neural import NueralContJumpProcess
from datasets import SyntheticDataset

from torchdiffeq import odeint

from utils.train_utils import (
    EarlyStopper, save_dynamic_model, load_dynamic_model,
    save_policy_model, load_policy_model, init_before_training
)

from utils.helpers import add_temporal_noise, event2count

from train import tau_schedule, set_decay_learning_rate
from env_mhp import MHPEnvNetint

from typing import List

def linear_annealing(current_step, total_steps, T_initial, T_final):
    fraction = min(current_step / total_steps, 1.0)
    return T_initial - fraction * (T_initial - T_final)


class CNJPTrainer:

    def __init__(
            self, 
            model : NueralContJumpProcess,
            train_config : dict,
            data_config : dict = None,
            eval_callback : bool = True,
        ) -> None:

        self.model = model


        self.model_lr = train_config.get('model_lr',1e-4)
        self.model_lr_decay = train_config.get("model_lr_decay",0.999)
        self.pem_lr = train_config.get('pem_lr',5e-4)
        self.pg_lr = train_config.get('pg_lr',1e-4)
        self.pg_lr_decay = train_config.get('pg_lr_decay', 0.93)

        self.horizon = train_config.get('horizon', 100)

        self.perminv_coeff = train_config.get("perminv_coeff",0.0)
        self.model_learning = train_config.get("model_learning", False)

        self.model_learning_experiment_dir = train_config.get('model_learning_experiment_dir','./model_learning')
        self.model_learning_experiment_name = train_config.get('model_learning_experiment_name','test')

        self.rl_experiment_dir = train_config.get("rl_experiment_dir", "./rl_training")
        self.rl_experiment_name = train_config.get("rl_experiment_name", "test")

        self.tau = 1.0
        self.decay_epo=  150
        self.warmup_epo = 10
        self.save_every = 10
        self.device = train_config.get("device", "cuda:0")
        self.eval_callback = eval_callback
        
        self._init_optimizers()
        self._init_loss_meter()

    def _init_optimizers(self):

        self.optimizer = torch.optim.AdamW([
                    {"params":self.model.ode_solver.func.state_cfn.parameters(), "lr": self.model_lr, "weight_decay":0.1},
                    {"params":self.model.ode_solver.func.state_dfn.parameters(), "lr": self.model_lr,"weight_decay":0.1},
                    {"params":self.model.ode_solver.func.intensity_fn.parameters(), "lr": self.model_lr,"weight_decay":0.1},
                    {"params":self.model._init_hstate, "lr" : 1e-3, "weight_decay":0.0},
                ])

        self.pg_optimizer = torch.optim.AdamW([
                            {"params":self.model.ode_solver.func.policy_fn.parameters(), "lr": self.pg_lr},
                            # {"params":self.model.ode_solver.func.represent_fn.parameters(), "lr": self.pg_lr},
                                ])
        self.rep_optimizer = torch.optim.AdamW(
            [
                {"params":self.model.ode_solver.func.represent_fn.parameters(), "lr": self.pg_lr},
            ]
        )

        self.pem_optimizer = torch.optim.AdamW([
                                    {"params":self.model.pemvalue.parameters(),"lr":self.pem_lr}
                                    ])
        
        self.scheduler1 = ExponentialLR(self.optimizer, gamma=self.model_lr_decay)

        # self.scheduler2 = CyclicCosineDecayLR(self.pg_optimizer, init_decay_epochs=self.decay_epo-self.warmup_epo,
        #                              min_decay_lr=1e-7,restart_interval=10,
        #                              restart_interval_multiplier=1.5,
        #                             restart_lr=5e-6,warmup_epochs=self.warmup_epo if self.warmup_epo != 0 else None,
        #                              warmup_start_lr=1e-6)

    def _init_loss_meter(self):
        self.loss_meter = logging_utils.RunningAverageMeter(0.98)
        self.loss_meter_policy = logging_utils.RunningAverageMeter(0.98)
        self.loss_meter_rep = logging_utils.RunningAverageMeter(0.98)
        self.loss_meter_pem = logging_utils.RunningAverageMeter(0.98)
        self.grad_meter = logging_utils.RunningAverageMeter(0.98)
        self.grad_meter_policy = logging_utils.RunningAverageMeter(0.98)

    def _init_before_model_learning(self):
        """Call before model training"""
        os.makedirs(self.model_learning_experiment_dir, exist_ok=True)
        savepath = os.path.join(self.model_learning_experiment_dir,self.model_learning_experiment_name)
        self.model_learning_writer = SummaryWriter(os.path.join(savepath,"tb_logdir"))

        self.model_learning_iteration_counter = itertools.count(0)
        self.best_nll = np.inf
        self.best_ml_epo = 0

    def train_model(self, dataloader, num_epoches):
        self._init_before_model_learning()

        for i in range(num_epoches):
            self.train_model_one_epoch(dataloader)
            print(f"Epoch {i}, Avg NLL {self.loss_meter.avg:.3f}")
            self.model_learning_writer.add_scalar("nll/epoch", self.loss_meter.avg, i)
            self.scheduler1.step()

    def train_model_one_epoch(self, dataloader):
        for batch in dataloader:
            self.train_model_one_batch(batch)

    def train_model_one_batch(self,batch):
        self.model.train()
        itr = next(self.model_learning_iteration_counter)
        

        event_data, event_time = batch  # (N, T, D), (N,T)
        
        ## set t0 to be [0-1)
        t0 = torch.max(event_time[:,0]-1,
                        torch.zeros_like(event_time[:,0])).to(event_time)
        self.optimizer.zero_grad()

        nlogp = self.model(
                    event_data=event_data,
                    event_time=event_time, 
                    t0=t0,
                    action_mask=None,
                    policy_learning=False
                )

        loss =  nlogp.mean()           
        loss.backward()
        self.optimizer.step()

        self.loss_meter.update(loss.item())
        self.model_learning_writer.add_scalar("nll/step", loss.item(),itr)

    def _init_before_rl(self):
        """Call before rl"""
        os.makedirs(self.rl_experiment_dir, exist_ok=True)
        savepath = os.path.join(self.rl_experiment_dir,self.rl_experiment_name)
        self.rl_writer = SummaryWriter(os.path.join(savepath,"tb_logdir"))

        self.rl_iteration_counter = itertools.count(0)
        self.rl_itr = 0
        self.best_rl_rewards = -np.inf
        self.best_rl_epo = 0

    def _update_dataset_by_env(self, dataloader , history):
        """update the dataset using new information from env"""
        ## new observations
        count_arr = torch.Tensor(event2count(np.array(history)[:,:2]))           ## event[:,:2] remove mark col
        event_time = torch.Tensor(add_temporal_noise(np.array(range(0,len(count_arr))), dt=1))

        ### reflash the dataset
        dataloader.dataset.regenerate(observations=count_arr, event_time=event_time)

    def _update_dataset_by_model(self,dataloader, event_data, event_time):
        dataloader.dataset.regenerate(observations=event_data, event_time=event_time)

    
    def train_meta_rl(
      self,
      dataloaders : List[DataLoader],
      num_epoches : int,
      num_subepoches : int,
      envs        : List[MHPEnvNetint],
      model_paths : List[str],
      seed        : int,      
    ):
        self._init_before_rl()
        if self.model_learning:
            self._init_before_model_learning()

        self.total_itrs = num_epoches * num_subepoches *  max(len(dataloader) for dataloader in dataloaders)

        for i in range(num_epoches):
            ## sample an environment
            idx = np.random.choice(range(len(dataloaders)))
            
            env = envs[idx]
            dataloader = dataloaders[idx]
            path = model_paths[idx]

            ## change the influence graph
            self.load_dynamic_model(path)
            self._update_model_by_node_num(dataloader)
            
            ## learn the policy
            self._train_rl(
                dataloader,
                num_subepoches,
                env,
                id = f"Env{idx+1}",
                seed= seed,
                out_epo= i,
                num_inner_epochs=num_subepoches
            )
            set_decay_learning_rate(self.rep_optimizer,0.88)

            if i % self.save_every == 5:
                self.save_policy_model(os.path.join(
                    self.rl_experiment_dir,
                    self.rl_experiment_name,
                    "policy_progress.ckpt"
                    ))
                self.save_pem_model(os.path.join(
                    self.rl_experiment_dir,
                    self.rl_experiment_name,
                    "pem_progress.ckpt"
                    ))

    def _update_model_by_node_num(self,dataloader):
        new_node_nums = dataloader.dataset.num_nodes
        new_adj = dataloader.dataset.adjacency_matrix
        new_cost = dataloader.dataset.cost_matrix 
        self.model.update_nodes_nums_adjs_cost(new_node_nums, new_adj, new_cost)

    def train_rl(
            self, 
            dataloader, 
            num_epoches,
            env,
            window_size,
            T,
            seed,
            id = "env0"
        ):
        """
        Args:
            env: HawkesEnv to evaluate trained policy
            window_size: window_size of observation
            T: maximum inference time
        """
        self._init_before_rl()
        if self.model_learning:
            self._init_before_model_learning()

        self.total_itrs = num_epoches * (len(dataloader))

        self._train_rl(dataloader,num_epoches,env,seed, id=id)

    def _train_rl(
            self,
            dataloader,
            num_epoches,
            env,
            seed,
            id,
            out_epo = 1,
            num_inner_epochs = 0
        ):
        for i in range(num_epoches):
            cur_time = time.time()
            self.train_rl_one_epoch(dataloader=dataloader, id=id)

            if self.eval_callback == True:
                reward_list, dynamic_intensities, history, event = self.eval_rl(
                    env=env,
                    seed=seed
                )
                event_data, event_time = event
                cur_reward = np.sum(reward_list)
                self.rl_writer.add_scalar(f"{id}/ept_rew",cur_reward, i+out_epo*num_inner_epochs)
                self.rl_writer.add_scalar(f"{id}/intensity",np.mean(dynamic_intensities), i+out_epo*num_inner_epochs)

            if self.model_learning:
                event_data, event_time, actions_control, dynamic_intensities = self.eval()
                # self._update_dataset_by_env(dataloader, history)
                self._update_dataset_by_model(dataloader, event_data, event_time)
                self.train_model_one_epoch(dataloader)

            elapsed_time = time.time() - cur_time
            self.print_rl(elapsed_time=elapsed_time, epoch=i+out_epo*num_inner_epochs, id=id)

            ## single rl use this
            set_decay_learning_rate(self.pg_optimizer,self.pg_lr_decay)
            set_decay_learning_rate(self.rep_optimizer,self.pg_lr_decay)

            ## meta RL use this
            # self.scheduler2.step()
            # set_decay_learning_rate(self.pg_optimizer,0.992)
            ## decay the model when training rl
            set_decay_learning_rate(self.optimizer, 0.9)

            if i % self.save_every == 5:
                self.save_policy_model(os.path.join(
                    self.rl_experiment_dir,
                    self.rl_experiment_name,
                    "policy_progress.ckpt"
                    ))

    def train_rl_one_epoch(self, dataloader, id="env0"):
        for batch in dataloader:
            self.train_rl_one_batch(batch,id=id)
            if self.perminv_coeff > 0:
                self.train_perminv_one_batch(batch)

    def train_rl_one_batch(self, batch, id=0):
        self.model.train()
        _, event_time = batch  # (N, T, D), (N,T)
        
        ## set t0 to be [0-1)
        t0 = torch.max(
                event_time[:,0]-1,
                torch.zeros_like(event_time[:,0])
                ).to(event_time)

        self.pg_optimizer.zero_grad()
        self.rep_optimizer.zero_grad()
        self.rl_itr = next(self.rl_iteration_counter)
        # self.tau = tau_schedule(self.rl_itr, self.total_itrs)
        self.tau = linear_annealing(
                current_step=self.rl_itr, 
                total_steps=self.total_itrs, 
                T_initial = 1.0, 
                T_final = 0.01
                )
        
        ## train model-based rl 
        pg_loss, rep_loss, _ , log_cost2go = self.model(
                            event_data=None,
                            event_time=event_time, 
                            t0=t0,
                            action_mask=None,
                            tau = self.tau,
                            policy_learning=True
                            )
        
        total_loss = pg_loss + self.perminv_coeff * rep_loss
        # print(rep_loss.item())
        ## cal grads respect to policy and val net
        
        # print(f"Rep: {rep_loss.item()}, PG: {pg_loss.item()}")
        policy_grads = torch.autograd.grad(total_loss,
                                            list(self.model.ode_solver.func.policy_fn.parameters()) + \
                                            list(self.model.ode_solver.func.represent_fn.parameters()),
                                            retain_graph=True)

        ## update gradients
        for grad, p in zip(policy_grads,
                            list(self.model.ode_solver.func.policy_fn.parameters()) + \
                                list(self.model.ode_solver.func.represent_fn.parameters())):
            # p.grad += grad
            p.grad = grad
        

        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                                        list(self.model.ode_solver.func.policy_fn.parameters()) + \
                                            list(self.model.ode_solver.func.represent_fn.parameters()), 
                                                max_norm=10.0).item()
        
        self.grad_meter_policy.update(grad_norm)
        
        ## detach loss to release cached graph every batch
        self.loss_meter_policy.update(pg_loss.detach())
        self.loss_meter_rep.update(rep_loss.detach())            
        self.pg_optimizer.step()
        self.rep_optimizer.step()

        self.rl_writer.add_scalar(f"{id}/pg_loss",pg_loss.item(), self.rl_itr)
        self.rl_writer.add_scalar(f"{id}/rep_loss",rep_loss.item(), self.rl_itr)
        self.rl_writer.add_scalar(f"{id}/reward_run_time",-torch.exp(log_cost2go.mean()).item(), self.rl_itr)


    def train_perminv_one_batch(self, batch):
        self.model.train()
        """Use dynamic programming to train perminv distance network"""

        event_data, event_time = batch  # (N, T, D), (N,T)
        
        ## set t0 to be [0-1)
        t0 = torch.max(
                event_time[:,0]-1,
                torch.zeros_like(event_time[:,0])
                ).to(event_time)
        
        _, _, pem_loss, _ = self.model(
                            event_data=None,
                            event_time=event_time, 
                            t0=t0,
                            action_mask=None,
                            tau = self.tau,
                            policy_learning=True
                            )

        self.pem_optimizer.zero_grad()
            # pem_loss = pem_loss.mean()
        pem_grads = torch.autograd.grad(pem_loss, self.model.pemvalue.parameters(),retain_graph=True)
        for grad, p in zip(pem_grads, self.model.pemvalue.parameters()):
            p.grad = grad
    
        self.pem_optimizer.step()
        
        self.loss_meter_pem.update(pem_loss.detach())
        self.model.update_target_pemvaluenet()

        self.rl_writer.add_scalar(f"pem_loss",pem_loss.item(), self.rl_itr)

    def eval_action(self, actions, mhp_env : MHPEnvNetint, seed=2):
        done = False
        mhp_env.reset(seed)
        reward_list = []
        while not done:
            cur_time = int(mhp_env.data[-1][0])
            action_index = cur_time - mhp_env.window_size
            if action_index >= 0:
                action = actions[action_index].cpu().numpy()
            else:
                ## no intervention
                action = np.zeros(shape=(mhp_env.dim * (mhp_env.dim - 1)))
            obs, reward, done, truncate, info = mhp_env.step(action)

            reward_list.append(reward)

        return reward_list, mhp_env.data

    def eval(self):
        self.model.eval()
        (event_data,event_time, _), actions_control, dynamic_intensities = self.model(
                                                event_data=None,
                                                event_time=None, 
                                                t0=0,
                                                t1=self.horizon,
                                                action_mask=None,
                                                inference=True,
                                                policy_learning = True
                                            )
        return event_data, event_time, actions_control, dynamic_intensities

    def eval_rl(self,env, seed=2):
        self.model.eval()
        # print("Eval model")
        event_data, event_time, actions_control, dynamic_intensities = self.eval()
        # print("Eval model end")    
        ## [0,:] extract from batch
        dynamic_intensities = dynamic_intensities[0,:].detach().cpu().numpy()
        # print('Eval action')
        reward_list, history = self.eval_action(
                    actions=actions_control['control_vec'],
                    mhp_env=env, 
                    seed=seed,
                    )
        # print("Finish Eval")
        return reward_list, dynamic_intensities, history, (event_data,event_time)
    
    # def save_dynamic_model(self,dir_name, name):
    #     save_dynamic_model(
    #         self.model, 
    #         self.optimizer, 
    #         dir_name,
    #         name
    #     )

        
    def print_rl(self, elapsed_time, epoch, id):
        print(
            f" Iter {self.rl_itr} | Epoch {epoch} | Time {elapsed_time:.1f} | {id} | DYNA LR {self.optimizer.param_groups[0]['lr']:.6f}"
            f" | PG LR {self.pg_optimizer.param_groups[0]['lr']:.6f}"
            f" | Tau {self.tau:.5f}"
            f" | Loss dyna {self.loss_meter.val if self.loss_meter.val is not None else 0. :.4f}({self.loss_meter.avg:.4f})"
            f" | Loss pg {self.loss_meter_policy.val if self.loss_meter_policy.val is not None else 0. :.4f}({self.loss_meter_policy.avg:.4f})"
            f" | Loss rep {self.loss_meter_rep.val if self.loss_meter_rep.val is not None else 0. :.4f}({self.loss_meter_rep.avg:.4f})"
            f" | Loss pem {self.loss_meter_pem.val if self.loss_meter_pem.val is not None else 0. :.4f}({self.loss_meter_pem.avg:.4f})"
            f" | GradNorm dyna {self.grad_meter.val if self.grad_meter.val is not None else 0. :.2f}({self.grad_meter.avg:.2f})"
            f" | GradNorm pg {self.grad_meter_policy.val if self.grad_meter_policy.val is not None else 0. :.2f}({self.grad_meter_policy.avg:.2f})"
        )

    def load_dynamic_model(self,path):
        ## map the loaded model to particular device
        checkpt = torch.load(path,map_location=self.device)
        self.model.ode_solver.func.state_cfn.load_state_dict(checkpt["cfn_state_dict"])
        self.model.ode_solver.func.state_dfn.load_state_dict(checkpt["dfn_state_dict"])
        self.model.ode_solver.func.intensity_fn.load_state_dict(checkpt["int_state_dict"])
        self.model.adjs.data = checkpt["adj_dict"]
        self.model._init_hstate.data = checkpt["init_state"]
        self.optimizer.load_state_dict(checkpt["optim_state_dict"])

    def load_policy_model(self,path):
        ## map the loaded model to particular device
        checkpt = torch.load(path,map_location=self.device)
        self.model.ode_solver.func.policy_fn.load_state_dict(checkpt["pol_state_dict"])
        self.model.ode_solver.func.represent_fn.load_state_dict(checkpt["rep_state_dict"])
        # self.pg_optimizer.load_state_dict(checkpt['optim_state_dict_pg'])
        # self.rep_optimizer.load_state_dict(checkpt['optim_state_dict_rep'])

    def load_pem_model(self, path):
        ## map the loaded model to particular device
        checkpt = torch.load(path,map_location=self.device)
        self.model.pemvalue.load_state_dict(checkpt["pem_state_dict"])
        self.model.target_pemvalue.load_state_dict(checkpt["target_pem_state_dict"])
        self.pem_optimizer.load_state_dict(checkpt['optim_state_dict'])
    
    def save_pem_model(self, path):
        torch.save(
            {
                "pem_state_dict": self.model.pemvalue.state_dict(),
                "target_pem_state_dict" : self.model.target_pemvalue.state_dict(),
                "optim_state_dict" : self.pem_optimizer.state_dict()
            },
            path
        )

    def save_policy_model(self,path):
        torch.save({
            # "state_dict": model.module.state_dict(),
            "rep_state_dict": self.model.ode_solver.func.represent_fn.state_dict(),
            "pol_state_dict": self.model.ode_solver.func.policy_fn.state_dict(),
            "optim_state_dict_pg": self.pg_optimizer.state_dict(),
            "optim_state_dict_rep" : self.rep_optimizer.state_dict(),
        }, path)

    def save_dynamic_model(self,path):
        torch.save({
            # "state_dict": model.module.state_dict(),
            "cfn_state_dict": self.model.ode_solver.func.state_cfn.state_dict(),
            "dfn_state_dict": self.model.ode_solver.func.state_dfn.state_dict(),
            "int_state_dict": self.model.ode_solver.func.intensity_fn.state_dict(),
            "adj_dict" : self.model.adjs.data,
            "init_state": self.model._init_hstate.data,
            "optim_state_dict": self.optimizer.state_dict(),
        }, path)


