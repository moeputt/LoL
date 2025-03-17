import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from torch.nn import functional as F
vmap_linear = torch.vmap(F.linear)
vmap_bmm_N0 = torch.vmap(torch.bmm, in_dims = (None, 0))
vmap_bmm_00 = torch.vmap(torch.bmm, in_dims = (0, 0))
vmap_linear_10 = torch.vmap(F.linear, in_dims = (1,0))
device = torch.device('cuda') 

class GLNonlin(nn.Module):
    def forward(self, u,v,eps=1e-6):
        to_norm = vmap_bmm_00(u, v.swapaxes(-1,-2))
        left_mul = F.relu(torch.sign((to_norm.sum(dim=3))).unsqueeze(-1))
        right_mul = F.relu(torch.sign((to_norm.sum(dim=2))).unsqueeze(-1))
        return left_mul * u, right_mul * v
        
class EquivariantLinear(nn.Module):
    def __init__(self, n, m, out_dim):
        super().__init__()
        self.w_1 = nn.Parameter(torch.randn(out_dim // 2, n)/(n)**.5)
        self.w_2 = nn.Parameter(torch.randn(out_dim // 2, m)/(m)**.5)
        
    def forward(self, u, v):
        uv = torch.cat([self.w_1 @ u, self.w_2 @v], dim = 1)
        return uv
        
        
        
class EquivariantMLP(nn.Module):
    def __init__(self, n, m, hidden_dim, n_layers, n_input_layers):
        super().__init__()
        self.n = n
        self.m = m
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_input_layers = n_input_layers
        self.equivariant_layers_u = nn.ParameterList()
        self.equivariant_layers_v = nn.ParameterList()
        self.first_layer = nn.ModuleList([EquivariantLinear(n, m, 2*hidden_dim) for n,m in zip(n, m)])
        
        for i in range(1, n_layers):
            layer_u = nn.Parameter(torch.randn(n_input_layers, self.hidden_dim, self.hidden_dim) / (self.hidden_dim)**.5)
            layer_v = nn.Parameter(torch.randn(n_input_layers, self.hidden_dim, self.hidden_dim) / (self.hidden_dim)**.5)
            
            self.equivariant_layers_u.append(layer_u)
            self.equivariant_layers_v.append(layer_v)
            
        self.nonlins = nn.ModuleList([GLNonlin() for i in range(n_layers - 1)])

    def forward(self,uvs): #list of tuples (u,v), each tuple is shape b x n x r. There are l tuples
        """Forward method of EquivariantMLP. This method first applies a separate equivariant linear transform to each """
        new_list = []
        for j, (u,v) in enumerate(uvs):
            out = self.first_layer[j](u,v)
            u, v = torch.split(out, self.hidden_dim, dim = 1)
            new_list.append((u,v))
            
        us = [uv[0] for uv in new_list]
        us = torch.stack(us,dim = 1)
        vs = [uv[1] for uv in new_list]
        vs = torch.stack(vs,dim = 1)
        
        for i in range(self.n_layers - 1):
            us_1, vs_1 = self.nonlins[i](us, vs)
            us_1 = vmap_bmm_N0(self.equivariant_layers_u[i], us_1)
            vs_1 = vmap_bmm_N0(self.equivariant_layers_v[i], vs_1)
            us = us_1
            vs = vs_1
            
        new_list = []
        for i in range(len(us[0])):
            new_list.append((us[:,i,...], vs[:,i,...]))
        uvs = new_list
        return new_list

class GLInvariantMLP(nn.Module):
    def __init__(self,ns, ms, n_input_layers,  out_dim,hidden_dim_equiv = 128, n_layers = 0, hidden_dim_inv = 128, clip = False):
        super().__init__()
        
        self.equiv_mlp = EquivariantMLP(ns, ms, hidden_dim_equiv, n_layers, n_input_layers)
        new_ns, new_ms = [hidden_dim_equiv for i in range(n_input_layers)], [hidden_dim_equiv for i in range(n_input_layers)]
        self.invariant_head = InvariantHead(new_ns, new_ms, hidden_dim_inv, out_dim, n_input_layers, clip = clip)
    def forward(self, uvs):
        uvs_1 = self.equiv_mlp(uvs)
        uvs_2 = self.invariant_head(uvs_1)
        return uvs_2
class InvariantHead(nn.Module):
    def __init__(self, ns, ms, hidden_dim, out_dim, n_input_layers, clip = False):
        super().__init__()
        self.ns = ns
        self.ms = ms
        self.invariant_outputs = nn.ModuleList([InvariantOutput(n, m, clip = clip) for n, m in zip(ns, ms)])
        total_length = sum([output.num_elem for output in self.invariant_outputs])
        self.linear1 = nn.Linear(total_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, uvs):
        out = torch.cat([f(u,v).flatten(start_dim=1) for (u,v), f in zip(uvs, self.invariant_outputs)],dim=1)
        out = self.linear1(out)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out
class SimpleNet(nn.Module):
    def __init__(self, ns, ms, hidden_dim, out_dim):
        super().__init__()
        self.ns = ns
        self.ms = ms
        self.lin_1 = nn.Linear((sum(ns) + sum(ms)), hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, out_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
    def forward(self, uvs):
        tot = []
        for j, (u,v) in enumerate(uvs):
            tot.append(u)
            tot.append(v)
        tot = torch.cat(tot, dim = 1).flatten(start_dim = 1)
        tot = self.lin_1(tot)
        tot = self.ln1(tot)
        tot = F.relu(tot)
        return self.lin_2(tot)
class BaselineNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin_1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.lin_2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.lin_1(x)
        x = self.ln1(x)
        x = F.relu(x)
        return self.lin_2(x)
class InvariantOutput(nn.Module):
    def __init__(self, n,m, clip = False):
        super().__init__()
        self.clip = clip
        self.m = m
        self.n = n
        if not clip:
            self.num_elem = self.n * self.m
        else:
            self.num_elem = 1
    def forward(self, u, v):
        if self.clip:
            return (torch.bmm(u, v.swapaxes(-1,-2))).mean(dim = [-1,-2]).unsqueeze(-1)
        else:
            return torch.bmm(u, v.swapaxes(-1,-2))