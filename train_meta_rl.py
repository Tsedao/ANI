import os
os.environ['CUDA_VISABLE_DEVICES'] = "0"
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
from utils.visualizations import plot_intensity

from train import (
    tau_schedule, set_decay_learning_rate, learning_rate_schedule
)
from env_mhp import MHPEnvNetint


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Train Meta-RL model")
    parser.add_argument("--regions",type='+', help="A list of named regions")
    parser.add_argument("--model_learning", type='store_true', help="adapative learning model")
    parser.add_argument("--gpu",type=int,default=0)

    args = parser.parse_args()

    # dim = 5
    # graph_type = 'circle'
    batch_size = 16
    seed = 2
    hstate_dim = 16
    hidden_dims = [64,64,64]
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    window_size = 7

    ## list all the config name
    # config_name = [f"config_c_{i}_ppo_1111" for i in range(0,6)]
    # config_name = [f"sumo_dim{i}" for i in [6,8,12,16]]
    config_name = args.regions

    ## dynamic path
    dyna_checkpt_paths = [
                        f"synthetic_data_model/{name}.ckpt" 
                        for name in config_name
                        ]
    ## model path
    data_paths = [
            f'synthetic_data/{name}'
            for name in config_name
            ]
    # env_config_path = 'configs/config_v1.yaml'
    model_learning = args.model_learning

    datasets = [
        SyntheticDataset(
        window_size=window_size, 
        data_path=data_path,
        max_nodes=25,
        device=device
        )
        for data_path in data_paths
    ]
    # data_event = dataset.events
    num_nodes = 25
    T = 100
    controls_limit = 25
    closure_limit = 3
    intervention_coeff = 0
    base_lr = 1e-5
    pg_lr = 5e-4
    pg_lr_decay = 0.96
    pem_lr = 5e-5
    contrastive_temperature = 0.1
    num_policy_layers = 3
    num_rep_layers = 3

    smooth_coeff = 0
    perminv_coeff = 1.0
    ode_solver = "dopri5"
    # ode_solver = "rk4"

    train_datas = [
            DataLoader(
              dataset, 
              batch_size=batch_size, 
              shuffle=True
              )
        for dataset in datasets
    ]

    init_before_training(seed=seed)


    env_configs = []
    for data_path in data_paths:
        with open(os.path.join(data_path,"env_params.json"),'rb') as f:
            env_config = json.load(f)
        env_configs.append(env_config)
   
    mhp_envs = [MHPEnvNetint(config=env_config) for env_config in env_configs]

    model = NueralContJumpProcess(
                  h_dims=hstate_dim,
                  hidden_dims=hidden_dims,
                  num_nodes=num_nodes,
                  total_time = T,
                  controls_limit= controls_limit,
                  closure_limit = closure_limit,
                  cost_matrix = datasets[0].cost_matrix,                          # temp assign
                  intervention_coeff = intervention_coeff,
                  smooth_coeff = smooth_coeff,
                  perminv_coeff = perminv_coeff,
                  ode_solver = ode_solver,
                  num_policy_layers = num_policy_layers,
                  num_rep_layers = num_rep_layers,
                  contrastive_temperature = contrastive_temperature,
                  adjs= datasets[0].adjacency_matrix                              # temp assign
            ).to(device)
    
    

    # checkpt = torch.load(dyna_checkpt_path, device)
    # # model.module.load_state_dict(checkpt["state_dict"])
    # load_dynamic_model(model, checkpt)

    experiment_path = 'experiments_rl_meta'
    experiment_id = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = (
                    f"{config_name[0].split('_')[0]}_"
                    f"model_lr_{base_lr}_"
                    f"pg_lr_{pg_lr}_"
                    f"pg_lr_d_{pg_lr_decay}_"
                    f"smooth_coeff_{smooth_coeff}_"
                    f"perminv_coeff_{perminv_coeff}_"
                    f"control_limit_{controls_limit}_"
                    f"ml_{model_learning}_"
                    f"bs_{batch_size}_"
                    f"ws_{window_size}_"
                    f"pl_{num_policy_layers}_"
                    f"rp_{num_rep_layers}_"
                    f"temp_{contrastive_temperature}_"
                    f"{ode_solver}_"
                    # f"rep_0.1_"                # rep learning rate : pg learning rate ratio
                    f"seed_{seed}_"
                    f"{experiment_id}"
                    )
    
    train_config = {
        "device" : device,
        "model_lr" : base_lr,
        "model_lr_decay" : 0.99,
        "model_learning": model_learning,
        "pg_lr" : pg_lr,
        "pem_lr" : pem_lr,
        "temperature" : contrastive_temperature,
        "pg_lr_decay" : pg_lr_decay,
        "perminv_coeff": perminv_coeff,
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
            eval_callback=False,
            )
    
    print("Start training Meta RL model")
    trainer.train_meta_rl(
        dataloaders=train_datas,
        num_epoches=80,
        envs=mhp_envs,
        model_paths=dyna_checkpt_paths,
        seed=seed,
        num_subepoches=5,
    )
    
    policy_folder = "synthetic_policy_model"
    os.makedirs(policy_folder, exist_ok=True)
    trainer.save_policy_model(
            path=os.path.join(
               policy_folder,
            f"{experiment_name}.ckpt")
        )
    pem_folder = "synthetic_pem_model"
    os.makedirs(pem_folder, exist_ok=True)
    trainer.save_pem_model(
            path=os.path.join(
               policy_folder,
            f"{experiment_name}.ckpt")
        )