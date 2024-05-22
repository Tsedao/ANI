import os
import math
import time
import yaml
import json
import itertools
import warnings
import torch.nn as nn
import random
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import logging_utils

from neural import NueralContJumpProcess
from trainer import CNJPTrainer
from datasets import SyntheticDataset

from utils.train_utils import (
    EarlyStopper, save_dynamic_model, load_dynamic_model,
    save_policy_model, load_policy_model, init_before_training
)

from utils.helpers import add_temporal_noise, event2count

from train import (
    tau_schedule, set_decay_learning_rate, learning_rate_schedule
)
from env_mhp import MHPEnvNetint


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Train RL")
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--seed",type=int,default=0)

    args = parser.parse_args()

    batch_size = 16
    seed = args.seed
    hstate_dim = 16
    hidden_dims = [64,64,64]
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    window_size = 7

    # dim = 20
    # graph_type = 'circle'
    # dyna_checkpt_path = f"model_pool/{graph_type}_{dim}.ckpt"
    # data_path = f'synthetic_data/{graph_type}_{dim}'

  
    data_path = args.data_path
    dyna_checkpt_path = f"synthetic_data_model/{data_path.replace('/','_')}.ckpt"
    data_path = f'synthetic_data/{data_path}'

    model_learning = False
    load_meta_policy = True
    train_eval = False

    dataset = SyntheticDataset(
        window_size=window_size, 
        data_path=data_path,
        max_nodes=25,
        device=device
    )
    # data_event = dataset.events
    num_nodes = dataset.num_nodes
    T = int(torch.max(dataset.dataset[:,-1]).item())
    controls_limit = 20
    closure_limit = 10
    intervention_coeff = 0
    base_lr = 1e-5
    # pg_lr = 1e-2    # small 5e-5 large 5e-4
    pg_lr = 1e-4
    pg_lr_decay = 0.9     # for meta policy please use small lr_decay [<0.9]
    # pg_lr_decay = 0.96     # train from scratch 
    pem_lr = 5e-4
    num_policy_layers = 3
    num_rep_layers = 3

    smooth_coeff = 0
    perminv_coeff = 0
    ode_solver = "dopri5"
    # ode_solver = "rk4"

    train_data = DataLoader(
              dataset, 
              batch_size=batch_size, 
              shuffle=True
              )
    init_before_training(seed=seed)

    if os.path.exists(os.path.join(data_path,"env_params.json")):
        with open(os.path.join(data_path,"env_params.json"),'rb') as f:
            env_config = json.load(f)
    
        mhp_env = MHPEnvNetint(config=env_config)
    else:
        ## covid and transportation does not simulation env
        mhp_env = None

    model = NueralContJumpProcess(
                  h_dims=hstate_dim,
                  hidden_dims=hidden_dims,
                  num_nodes=num_nodes,
                  total_time = T,
                  controls_limit= controls_limit,
                  closure_limit = closure_limit,
                  cost_matrix = dataset.cost_matrix,
                  intervention_coeff = intervention_coeff,
                  smooth_coeff = smooth_coeff,
                  perminv_coeff = perminv_coeff,
                  ode_solver = ode_solver,
                  num_policy_layers = num_policy_layers,
                  num_rep_layers = num_rep_layers,
                  adjs= dataset.adjacency_matrix
            ).to(device)
    
    

    # checkpt = torch.load(dyna_checkpt_path, device)
    # model.module.load_state_dict(checkpt["state_dict"])
    # load_dynamic_model(model, checkpt)

    experiment_path = 'experiments_rl'
    experiment_id = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = (
                    # f"{graph_type}_{dim}_"
                    f"{data_path.replace('/','_')}_"
                    f"model_lr_{base_lr}_"
                    f"pg_lr_{pg_lr}_"
                    f"smooth_coeff_{smooth_coeff}_"
                    f"perminv_coeff_{perminv_coeff}_"
                    f"control_limit_{controls_limit}_"
                    f"ml_{model_learning}_"
                    f"bs_{batch_size}_"
                    f"ws_{window_size}_"
                    f"pl_{num_policy_layers}_"
                    f"rp_{num_rep_layers}_"
                    f"meta_{load_meta_policy}_"
                    f"{ode_solver}_"
                    f"seed_{seed}_"
                    f"{experiment_id}"
                    )
    
    train_config = {
        "device" : device,
        "model_lr" : base_lr,
        "model_lr_decay" : 0.99,
        "horizon" : T,
        "model_learning": model_learning,
        "pg_lr" : pg_lr,
        "pg_lr_decay" : pg_lr_decay,
        "pem_lr" : pem_lr,
        "perminv_coeff": 0.0,
        "rl_experiment_dir": experiment_path,
        "rl_experiment_name":experiment_name
    }

    data_config = {
        "window_size" : window_size,
        "T"           : T
    }

    trainer = CNJPTrainer(
            model,
            train_config=train_config,
            data_config=data_config,
            eval_callback=train_eval,
            )
    ## loading dynamic model
    try:
        trainer.load_dynamic_model(path=dyna_checkpt_path)
        print(f"Successfully loading dynamic from {dyna_checkpt_path}...")
    except RuntimeError:
        warnings.warn("Random Intialize dynamic model", category=RuntimeWarning)
   
    policy_path = "synthetic_data_meta_policy"
    meta_policy_model_path = os.path.join(policy_path, "policy_progress.ckpt")
    
    if load_meta_policy:
        try:
            trainer.load_policy_model(meta_policy_model_path)
            print(f"Successfully loading policy from {meta_policy_model_path}...")
        except RuntimeError:
            warnings.warn("Random Intialize policy model", category=RuntimeWarning)
        
    
    print("Start RL ...")
    trainer.train_rl(
        dataloader=train_data,
        num_epoches=100,
        window_size=window_size,
        env=mhp_env,
        T=T,
        seed=seed
    )

    policy_folder = "synthetic_policy_model"
    os.makedirs(policy_folder, exist_ok=True)
    trainer.save_policy_model(
            path=os.path.join(
               policy_folder,
            f"{args.data_path.replace('/','_')}_{seed}.ckpt")
        )

    action_path = "action_vis"
    os.makedirs(action_path, exist_ok=True)
    action_npy_path = os.path.join(action_path, f"{args.data_path.replace('/','_')}.npy")
    event_data, event_time, actions_control, dynamic_intensities = trainer.eval()
    actions = torch.cat(actions_control['control'],dim=1).cpu().numpy()
    np.save(action_npy_path,actions)