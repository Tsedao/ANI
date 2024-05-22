import torch
import math

import torch.nn as nn
import torch.nn.functional as F

import layers.diffeq_layers as diffeq_layers

class ActNorm(nn.Module):

    def __init__(self, num_features, init_scale=1.0):
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.init_scale = init_scale
        self.register_buffer('initialized', torch.tensor(0))

    def forward(self, x):
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_ = x.reshape(-1, x.shape[-1])
                batch_mean = torch.mean(x_, dim=0)
                
                # for numerical issues
                batch_var = torch.var(x_, dim=0,unbiased=False)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))
                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var) + math.log(self.init_scale))
                self.initialized.fill_(1)

        bias = self.bias.expand_as(x)
        weight = self.weight.expand_as(x)
        # y = (x + bias) * torch.exp(weight)
        y = (x + bias) * F.softplus(weight)
        return y

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))
    
    
class Sine(nn.Module):

    def forward(self, x):
        return torch.sin(x)


class Swish(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5] * dim))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta)))

    def extra_repr(self):
        return f'{self.beta.nelement()}'


class GatedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.gate = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x) * torch.sigmoid(self.gate(x))
    
def construct_diffeqnet(
            input_dim, 
            hidden_dims, 
            output_dim,
            num_nodes,
            max_nodes = 25,
            time_dependent=False, 
            actfn="softplus", 
            zero_init=False, 
            gated=False
    ):

    linear_fn = diffeq_layers.IgnoreLinear if not time_dependent else diffeq_layers.ConcatLinear_v2
    bias_fn = diffeq_layers.NodesBias
    
    if gated:
        linear_fn = GatedLinear

    layers = []
    if len(hidden_dims) > 0:
        dims = [input_dim] + list(hidden_dims)
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(linear_fn(d_in, d_out))
            layers.append(bias_fn(num_nodes, max_nodes,d_out))
            layers.append(ActNorm(d_out))
            if not gated:
                layers.append(ACTFNS[actfn](d_out))
        layers.append(linear_fn(hidden_dims[-1], output_dim))
    else:
        layers.append(linear_fn(input_dim, output_dim))

    # Initialize to zero.
    if zero_init:
        for m in layers[-1].modules():
            if isinstance(m, nn.Linear):
                m.weight.data.fill_(0)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    diffeqnet = diffeq_layers.SequentialDiffEq(*layers)

    return diffeqnet


ACTFNS = {
    "softplus": lambda dim: diffeq_layers.diffeq_wrapper(nn.Softplus()),
    "swish": lambda dim: diffeq_layers.diffeq_wrapper(Swish(dim)),
    "celu": lambda dim: diffeq_layers.diffeq_wrapper(nn.CELU()),
    "relu": lambda dim: diffeq_layers.diffeq_wrapper(nn.ReLU(inplace=True))
}

class GATLayers(nn.Module):

    def __init__(
            self, 
            c_in, 
            c_out, 
            # num_heads=1,
            num_nodes=4,
            hidden_dims=[32],
            # concat_heads=True, 
            alpha=0.2
        ):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            hidden_dims - Dimensionalities of middleware layers
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        super().__init__()
        # self.num_heads = num_heads
        # self.concat_heads = concat_heads
        # if self.concat_heads:
        #     assert c_out % num_heads == 0, "Number of output features must be a multiple of the count of heads."
        #     c_out = c_out // num_heads

        # Sub-modules and parameters needed in the layer
        # self.projection = nn.Linear(c_in, c_out * num_heads)
        
        # self.projection = construct_diffeqnet(input_dim=c_in+1, 
        #                                       hidden_dims=hidden_dims, 
        #                                       output_dim=c_out * num_heads, 
        #                                       time_dependent=False, 
        #                                       actfn='softplus', zero_init=False)
        
#         self.projection = nn.ModuleList([nn.GRUCell(input_size=1,
#                                     hidden_size=c_in) for i in range(num_nodes)])
        
        
#         self.time_projection = nn.ModuleList([nn.Linear(1,c_in,bias=False) for _ in range(num_nodes)])
#         self.actnorms = nn.ModuleList([ActNorm(c_in) for _ in range(num_nodes)])
#         self.time_scale = nn.ModuleList([nn.Linear(1, c_in) for _ in range(num_nodes)])
        
#         [self.time_projection[i].weight.data.fill_(0.0) for i in range(num_nodes)]
        
        self.projection = nn.GRUCell(input_size=1,
                                    hidden_size=c_in)
        
        
        self.time_projection = nn.Linear(1,c_in,bias=False)
        self.actnorms = ActNorm(c_in)
        # self.time_scale = nn.Linear(1, c_in)    
        self.time_projection.weight.data.fill_(0.0)
 
        
#         for name, p in self.projection.named_parameters():
#             p.data.fill_(0)
        
        # self.a = nn.Parameter(torch.Tensor(num_heads, 2 * c_out)) # One per head
        # self.activation = nn.LeakyReLU(alpha)
        # self.activation = nn.Tanh()

        # Initialization from the original implementation
        # self.projection.layers.apply(lambda x : init_weights(x, gain=1.414))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, t, node_feats, adj_matrix, x_obs):
        """
        Inputs:
            t - Input time. Shape: [batch_size, ]
            node_feats - Input features of the node. Shape: [batch_size,num_nodes, c_in]
            x_obs - Input observations. Shape: [batch_size, num_nodes]
            adj_matrix - Adjacency matrix including self-connections. Shape: [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes = node_feats.size(0), node_feats.size(1)
        
        
        # node_feats_ = torch.cat([node_feats, torch.log(torch.abs(x_obs.unsqueeze(2))+1)],dim=2)
        # # Apply linear layer and sort nodes by head
        # node_feats_ = self.projection(t, node_feats_)
        
        new_node_feats = []
        # t_proj = self.time_projection(t.view(-1,1)).unsqueeze(1)
        for i in range(num_nodes):
            # t_scale = self.time_scale(t.view(-1, 1)).unsqueeze(1)
            node_feats_normed = self.actnorms(node_feats[:,i,:])
            # node_feats_normed = F.normalize(node_feats[:,i,:],dim=1)
            n_proj = self.projection(torch.log(torch.abs(x_obs[:,i:i+1])+1.0), 
                                 node_feats_normed).unsqueeze(1)
            proj = n_proj
            new_node_feats.append(proj)
        node_feats_ = torch.cat(new_node_feats,dim=1)
        
        # print("node_feats",node_feats_.dtype)
        # print("adj_matrix", adj_matrix.dtype)
        node_feats_ = node_feats_.view(batch_size, num_nodes, -1)
        node_feats_delta = torch.bmm(adj_matrix, node_feats_)
        # print("delta:",node_feats_delta)
        node_feats = node_feats_delta

        return node_feats
    
    
class DiffLayers(nn.Module):
    """Continuous Evolution"""
    def __init__(self, c_in, hidden_dims, num_nodes, max_nodes=25,actfn='softplus'):
        super().__init__()
        # self.projections = nn.ModuleList([construct_diffeqnet(
        #                                           input_dim=c_in, 
        #                                           hidden_dims=hidden_dims, 
        #                                           output_dim=c_in, 
        #                                           time_dependent=True, 
        #                                           actfn=actfn, 
        #                                            zero_init=True
        #                                     ) for i in range(num_nodes)
        #                                 ])
        self.projection = construct_diffeqnet(
                                                  input_dim=c_in, 
                                                  hidden_dims=hidden_dims, 
                                                  output_dim=c_in,
                                                  num_nodes = num_nodes,
                                                  max_nodes = max_nodes,
                                                  time_dependent=True, 
                                                  actfn=actfn, 
                                                   zero_init=True
                                            )
    def forward(self, t, hstate):
        """
        Args:
            hstate : (N,D,H)
        """
#         bsz = hstate.shape[0]
        
#         dstate = torch.zeros_like(hstate)
        
#         for i, proj in enumerate(self.projections):
#             dstate[:,i,:] = proj(t,hstate[:,i,:])
        
        dstate = self.projection(t, hstate)
        return dstate