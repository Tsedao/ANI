import torch
import os
import json

import pandas as pd
import numpy as np

from configs import root_path

class COVIDDataset(torch.utils.data.Dataset):
    
    def __init__(self, horizon,device='cpu', state='New Jersey',split=0):
        
        data_path = os.path.join(root_path,f"data/covid19/{state}/split_{split}")
        df = pd.read_csv(os.path.join(data_path,"covid.csv"),index_col=0)
        
        self.dataset = torch.Tensor(df.to_numpy()).to(device)
        self.scale = torch.max(self.dataset,dim=0)[0] - torch.min(self.dataset, dim=0)[0]
        self.cost_matrix = torch.Tensor(np.load(os.path.join(data_path,"cost_matrix.npy"))).to(device)
        self.mini_popu = np.load(os.path.join(data_path,"min_population.npy"))
        self.names = df.columns
        self.num_nodes = self.dataset.shape[-1] - 1
        self.window_size = horizon
        self.device=device
    
    def __len__(self, ):
        return len(self.dataset) - self.window_size + 1
    def __getitem__(self, index):
        return self.dataset[index:index+self.window_size,:-1],self.dataset[index:index+self.window_size,-1]
    
    def regenerate(self, observations, event_time,device=None):
        new_data = torch.cat([observations, event_time.unsqueeze(-1)],dim=-1).to(self.device if not device else device)
        
        self.dataset = new_data

class SyntheticDataset(torch.utils.data.Dataset):
   
    def __init__(self, window_size, data_path, max_nodes=25, device='cpu'):
        super().__init__()
        self.dataset = torch.Tensor(np.load(os.path.join(data_path,"train_count.npy"))).to(device)
        if os.path.exists(os.path.join(data_path,"train_event.npy")):
            self.events = np.load(os.path.join(data_path,"train_event.npy"))
        self.length = len(self.dataset)
        self.num_nodes = self.dataset.shape[-1] - 1
        self.max_nodes = max_nodes
        self.num_appends = self.max_nodes - self.num_nodes
        self.window_size = window_size
        self.device=device

    
        self._load_influence_matrix(data_path)
        self._load_region_property(data_path)   

        if self.num_appends > 0:
            self.baseline = self.padding_baseline_zeros(self.baseline)
            self.adjacency_matrix = self.padding_adj_zeros(self.adjacency_matrix)
            # self.cost_matrix = self.padding_adj_zeros(self.cost_matrix)
            # self.mini_popu = self.padding_adj_zeros(self.mini_popu)
            self.dataset = self.padding_obs_zeros(self.dataset)

    def _load_influence_matrix(self, path):
        if os.path.exists(os.path.join(path,"env_params.json")):
            with open(os.path.join(path,"env_params.json"), "r") as f:
                env_params = json.load(f)
                self.adjacency_matrix = torch.Tensor(env_params['adjacency']).to(self.device)
                self.baseline = torch.Tensor(env_params['baseline']).to(self.device)
                self.omege = env_params['omega']
        else:
            self.adjacency_matrix = torch.ones(size=(self.num_nodes,self.num_nodes)).to(self.device)
            self.baseline = torch.zeros(size = (self.num_nodes,)).to(self.device)
            self.omege = 4.0
    
    def _load_region_property(self, path):
        if os.path.exists(os.path.join(path,"cost_matrix.npy")):
            self.cost_matrix = torch.Tensor(np.load(os.path.join(path,"cost_matrix.npy"))).to(self.device)
        else:
            self.cost_matrix = torch.zeros(size=(self.num_nodes,self.num_nodes)).to(self.device)
        if os.path.exists(os.path.join(path, "min_population.npy")):
            self.mini_popu = torch.Tensor(np.load(os.path.join(path,"min_population.npy"))).to(self.device)
        else:
            self.mini_popu = torch.zeros(size=(self.num_nodes,self.num_nodes)).to(device)
        

    def padding_obs_zeros(self, obs):
        paddings = torch.zeros(size=(self.length, self.max_nodes)).to(self.device)
        paddings[:,:self.num_nodes] = obs[:,:self.num_nodes] # nodes 
        paddings[:,-1] = obs[:,-1]                           # time
        return paddings
    
    def padding_adj_zeros(self,adj):
        paddings = torch.zeros(size=(self.max_nodes, self.max_nodes)).to(adj)
        paddings[:self.num_nodes, :self.num_nodes] = adj
        return paddings
    
    def padding_baseline_zeros(self, baseline):
        paddings = torch.zeros(size=(self.max_nodes,)).to(baseline)
        paddings[:self.num_nodes] = baseline
        return paddings


    def __len__(self, ):
        return len(self.dataset) - self.window_size + 1
    def __getitem__(self, index):
        return self.dataset[index:index+self.window_size,:-1],self.dataset[index:index+self.window_size,-1]
    

    def regenerate(self, observations, event_time,device=None):
        new_data = torch.cat([observations, event_time.unsqueeze(-1)],dim=-1).to(self.device if not device else device)
        
        self.dataset = new_data