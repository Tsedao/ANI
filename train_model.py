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

import matplotlib.pyplot as plt
import logging_utils

from neural import NueralContJumpProcess
from configs import str2bool
from datasets import SyntheticDataset

from torchdiffeq import odeint

from utils.train_utils import (
    EarlyStopper, save_dynamic_model, load_dynamic_model,
    save_policy_model, load_policy_model, init_before_training
)

from utils.visualizations import plot_intensity

from utils.helpers import add_temporal_noise, event2count

from train import tau_schedule, set_decay_learning_rate
from trainer import CNJPTrainer

import argparse

if  __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--data_path",type=str)
    parser.add_argument("--gpu",type=int,default=0)
    parser.add_argument("--learning_influence_matrix",type='store_true')
    parser.add_argument("--resume", help="whether restore model",
                    action="store_false")
    parser.add_argument("-e", "--epoches",type=int,default=300)

    args = parser.parse_args()

    batch_size = 16                 ## relatively good 
    hstate_dim = 16
    hidden_dims = [64,64,64]
    seed = 2048
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    data_path = args.data_path

    dataset = SyntheticDataset(
                        window_size=7, 
                        data_path=os.path.join(
                                "synthetic_data",data_path),
                        device=device,
                        max_nodes=25
                                )
    num_nodes = dataset.num_nodes
    T = int(torch.max(dataset.dataset[:,-1]).item())
    controls_limit = 0
    closure_limit = 3
    intervention_coeff = 0

    ### synthetic parameters
    # base_lr = 0.01

    ## sumo parameters
    # base_lr = 0.001

    ## covid parameters
    base_lr = 0.01

    lr_decay = 0.999
    smooth_coeff = 0
    perminv_coeff = 0
    # ode_solver = "rk4"
    ode_solver = "dopri5"

    train_data = DataLoader(
              dataset, 
              batch_size=batch_size, 
              shuffle=True
              )
    
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
                  adjs= dataset.adjacency_matrix if not args.learning_influence_matrix else None
            ).to(device)
    
    experiment_path = 'experiments_model_learning'

    experiment_id = time.strftime("%Y%m%d_%H%M%S")
    experiment_name = (
                    f"{data_path}_"
                    f"model_lr_{base_lr}_"
                    f"decay_rate_{lr_decay}_"
                    f"bs_{batch_size}_"
                    f"hstate_dim_{hstate_dim}_"
                    f"hidden_dims_{hidden_dims}_"
                    f"seed_{seed}_"
                    f"{experiment_id}"
                    )
    
    train_config = {
        "device" : device,
        "model_lr" : base_lr,
        "model_lr_decay" : lr_decay,
        "pg_lr" : 1e-4,
        "model_learning_experiment_dir": experiment_path,
        "model_learning_experiment_name":experiment_name
    }
    init_before_training(seed=seed)
    trainer = CNJPTrainer(
            model,
            train_config=train_config
            )

    model_folder = "synthetic_data_model"

    model_path = os.path.join(
            model_folder,
            f"{data_path.replace('/', '_')}.ckpt"
        )

    if os.path.exists(data_path.replace('/', '_')):
        print(f"Found model in {data_path.replace('/', '_')}, loading exists model ...")
        trainer.load_dynamic_model(
            path = model_path
        )
    if args.resume:
        print("Start training model ...")
        trainer.train_model(dataloader=train_data,num_epoches=args.epoches)
    
        os.makedirs(model_folder, exist_ok=True)
        trainer.save_dynamic_model(
                path=os.path.join(
                    model_folder,
                f"{data_path.replace('/', '_')}.ckpt")
            )
    

    model_folder_vis = "synthetic_data_model_vis"
    os.makedirs(model_folder_vis, exist_ok=True)

    plot_intensity(dataset=dataset, model=model, save_path=
                   os.path.join(
                    model_folder_vis,
                    f"{data_path.replace('/', '_')}.pdf"
                )
            )