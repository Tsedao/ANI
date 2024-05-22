import torch

import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint


class IntensityNet(nn.Module):
    def __init__(self, hdim, num_nodes,num_layers=1, max_nodes=25):
        super().__init__()
        assert num_layers >= 1
        self.linears = nn.ModuleList([
            nn.Linear(hdim, hdim * 2),
            nn.ReLU()
        ])
        self.D = num_nodes
        # self.bias_1 = nn.Parameter(torch.Tensor(max_nodes))
        # self.bias_2 = nn.Parameter(torch.Tensor(max_nodes))
        for i in range(num_layers-1):
            self.linears.append(nn.Linear(hdim * 2, hdim * 2))
            self.linears.append(nn.Tanh())
        self.linears.append(nn.Linear(hdim*2, 1))
        # self.bias_1.data.fill_(0)
        # self.bias_2.data.fill_(0)

        for m in self.linears:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight,gain=2)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
        
    def forward(self, hstate):
        
        x_in = hstate ## [bs, N, d]
        for l in self.linears:
            x_in = l(x_in)

        x_in = x_in.squeeze(-1)
     
        x_out = F.softplus(x_in,beta=10) ## softplus r
        return x_out
    
    
class ValueNet(nn.Module):
    def __init__(
        self, hdim, num_nodes, num_layers=1
    ):
        super().__init__()
        assert num_layers >= 1
        self.linears = nn.ModuleList([
            nn.Linear(hdim, hdim * 4),
            nn.Softplus()
        ])
        self.linear_final = nn.Linear(num_nodes,1)
        self.act = nn.Softplus()
        for i in range(num_layers-1):
            self.linears.append(nn.Linear(hdim * 4, hdim * 4))
            self.linears.append(nn.Softplus())
        self.linears.append(nn.Linear(hdim*4, 1))
        
    def forward(self, hstate):
        
        # batch_size, _, _ = hstate.size()
        # x_in = hstate.reshape(batch_size,-1)
        x_in = hstate
        for l in self.linears:
            x_in = l(x_in)
        x_out = x_in.squeeze(-1)
        # x_out = -torch.exp(torch.sigmoid(x_in) * 10)
        # x_out = torch.tanh(x_out)
        # x_out = torch.sum(x_out, dim=-1)   # [N, D] -> [N,]
        x_out = self.linear_final(self.act(x_out)).squeeze(-1) # [N, D] -> [N,]
        return x_out
    
    

class PolicyNet(nn.Module):
    def __init__(
        self, hdim, num_nodes, num_dummy=3, num_layers=2, max_nodes=25,
    ):
        super().__init__()
        assert num_layers >= 1
        assert num_dummy <= num_nodes ** 2
        
        self.K = num_dummy
        self.D = num_nodes
        self.linears = nn.ModuleList([
            nn.Linear(64 * 2, hdim * 4),
            nn.Softplus()
        ])
        for i in range(num_layers-1):
            self.linears.append(nn.Linear(hdim * 4, hdim * 4))
            self.linears.append(nn.LeakyReLU())
        self.action_proj = nn.Linear(hdim * 4, (max_nodes * (max_nodes - 1)))
        self.dummy_proj = nn.Linear(hdim * 4, max_nodes * max_nodes)
        
#         for m in self.linears:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 # m.weight.data.fill_(0.01)
        
    def forward(self, pos_embed, mage_embed):
        
        embed = torch.cat([pos_embed,mage_embed], dim=-1)
        for l in self.linears:
            embed = l(embed)
            
        logits = self.action_proj(embed)[:,:(self.D * (self.D - 1))]
        dummy_logits = self.dummy_proj(embed)[:,:self.K]
        
        logits = torch.cat([logits, dummy_logits], dim = -1)
        return logits
    

    
class RepresentationNet(nn.Module):
    def __init__(
        self, hdim, num_nodes, embed_dim=64,
        num_layers=4, nhead=4,d_model=128,
    ):
        
        super().__init__()
        assert num_layers >= 1
       
        self.D = num_nodes
        
        self.proj_layer = nn.Linear(hdim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=hdim*8, 
                                                   dropout=0.6)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.pos_proj_layer_1 = nn.Linear(d_model, 1)
        # self.pos_proj_layer_2 = nn.Linear(self.D, embed_dim)
        
        # self.mage_proj_layer_1 = nn.Linear(d_model, 1)
        # self.mage_proj_layer_2 = nn.Linear(self.D, embed_dim)

        self.pos_proj_layer_1 = nn.Linear(d_model, hdim)
        self.pos_proj_layer_2 = nn.Linear(hdim, embed_dim)

        self.mage_proj_layer_1 = nn.Linear(d_model, hdim)
        self.mage_proj_layer_2 = nn.Linear(hdim, embed_dim)

        # self.dummy_proj_2 = nn.Linear(max_nodes, num_dummy)
    
    def forward(self, hstate):
        
        batch_size, _, _ = hstate.size()
        
        hstate = self.proj_layer(hstate)
        trans_out = self.transformer_encoder(hstate) # [B,D,d_models] -> [B,D,d_models]
        
        # pos_embed = F.leaky_relu(self.pos_proj_layer_1(trans_out)).reshape(batch_size,-1) # [B,D,d_models] -> [B,D]
        # pos_embed = self.pos_proj_layer_2(pos_embed)
        
        pos_embed = F.leaky_relu(self.pos_proj_layer_1(trans_out.mean(dim=1))) # [B,D,d_models] -> [B,d_models] -> [B,hdim]
        pos_embed = self.pos_proj_layer_2(pos_embed)

        
        # mage_embed = F.leaky_relu(self.mage_proj_layer_1(trans_out)).reshape(batch_size,-1) # [B,D,d_models] -> [B,D]
        # mage_embed = self.mage_proj_layer_2(mage_embed)

        mage_embed = F.leaky_relu(self.mage_proj_layer_1(trans_out.mean(dim=1)))
        mage_embed = self.mage_proj_layer_2(mage_embed)
     
        
        return pos_embed, mage_embed

class PolicyTransNet(nn.Module):
    def __init__(
        self, hdim, num_nodes, 
        num_dummy=3, num_layers=3, nhead=4, max_nodes=25,d_model=128,
    ):
        
        super().__init__()
        assert num_layers >= 1
        assert num_dummy <= num_nodes ** 2
        
        self.K = num_dummy
        self.D = num_nodes
        
        self.proj_layer = nn.Linear(hdim+num_nodes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=hdim*4, 
                                                   dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_proj = nn.Linear(d_model, max_nodes - 1)
        
        self.dummy_proj = nn.Linear(d_model, max_nodes)
        # self.dummy_proj_2 = nn.Linear(max_nodes, num_dummy)
    
    def forward(self, hstate, adj):
        
        batch_size, _, _ = hstate.size()
        
        hstate = torch.cat([hstate, adj], dim=-1)
        hstate = self.proj_layer(hstate)
        trans_out = self.transformer_encoder(hstate) # [B,D,H+D] -> [B,D,d_models]
        logits_out = self.policy_proj(trans_out)[:,:,:(self.D-1)]            # [B,D,d_models] -> [B,D,NUM_NODES-1]
        logits_out = logits_out.reshape(batch_size, self.D * (self.D - 1))
        
        dummy_out = F.relu(self.dummy_proj(trans_out))        # [B,D,H] -> [B,D,D]
        dummy_logits = dummy_out.reshape(batch_size,-1)[:,:self.K]             # [B,D,D] -> [B, NUM_DUMMY]
        
        logits = torch.cat([logits_out, dummy_logits], dim = -1)
        
        return logits
    
    
class ValueTransNet(nn.Module):
    def __init__(
        self, hdim, num_nodes,  num_layers=3, nhead=4,  max_nodes=25, d_model=128,
    ):
        
        super().__init__()
        assert num_layers >= 1
        
        self.D = num_nodes
        
        self.proj_layer = nn.Linear(hdim+num_nodes, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=nhead,
                                                   dim_feedforward=hdim*4, 
                                                   dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_layers)
        self.val_proj = nn.Linear(d_model, 1)
    
    
    def forward(self, hstate, adj):
        
        batch_size, _, _ = hstate.size()
        
        hstate = torch.cat([hstate, adj], dim=-1)
        hstate = self.proj_layer(hstate)
        trans_out = self.transformer_encoder(hstate) # [B,D,H] -> [B,D,d_models]
        val_out = self.val_proj(trans_out)            # [B,D,d_models] -> [B,D,1]
        val_out = val_out.reshape(batch_size, self.D)
        
        ## sum pooling
        val = torch.sum(val_out, dim=1)  # [B,D] -> [B,]
        
        return val
    
    

class PEMValueTransNet(nn.Module):
    def __init__(
        self, hdim, num_nodes,  num_layers=3, nhead=4,  max_nodes=25,
    ):
        
        super().__init__()
        assert num_layers >= 1
        
        self.D = num_nodes
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hdim*2, 
                                                   nhead=nhead,
                                                   dim_feedforward=hdim*4, 
                                                   dropout=0.5)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                         num_layers=num_layers)
        self.val_proj = nn.Linear(hdim*2, 1)
    
    
    def forward(self, hstate, perm_hstate):
        
        batch_size, _, _ = hstate.size()
        
        hstate = torch.cat([hstate, perm_hstate], dim=-1) # [B,D,H] -> [B,D,2*H]
        trans_out = self.transformer_encoder(hstate) 
        val_out = F.relu(self.val_proj(trans_out))            # [B,D,2*H] -> [B,D,1]
        val_out = val_out.reshape(batch_size, self.D)
        
        ## sum pooling
        val = torch.sum(val_out, dim=1)  # [B,D] -> [B,]
        
        return val
        
        
        
        
        
        
        