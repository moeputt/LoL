import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from torch.nn import functional as F
vmap_linear = torch.vmap(F.linear)
vmap_bmm_N0 = torch.vmap(torch.bmm, in_dims = (None, 0))
vmap_bmm_00 = torch.vmap(torch.bmm, in_dims = (0, 0))
def vmap_bmm_N0_new(A, B):
    return torch.einsum('lij,bljk->blik', A, B)
vmap_linear_10 = torch.vmap(F.linear, in_dims = (1,0))
device = torch.device('cuda')

class GLNonlinFast(nn.Module):
    def forward(self, uvs):
        u,v = torch.chunk(uvs, 2, dim = 1)
        vT = v.swapaxes(-1,-2)
        left_mul = F.relu(torch.sign(vmap_bmm_00(u, vT.sum(dim = -1).unsqueeze(-1))))
        right_mul = F.relu(torch.sign(vmap_bmm_00(u.sum(dim = -2).unsqueeze(-2), vT))).swapaxes(-1,-2)
        return torch.cat([left_mul * u, right_mul * v], dim = 1)
class GLNonlinSlow(nn.Module):
    def forward(self, uvs,eps=1e-6):
        #uvs is b x 2n x l x r
        u,v = torch.chunk(uvs, 2, dim = 1)
        to_norm = vmap_bmm_00(u, v.swapaxes(-1,-2))
        left_mul = F.relu(torch.sign((to_norm.sum(dim=3))).unsqueeze(-1))
        right_mul = F.relu(torch.sign((to_norm.sum(dim=2))).unsqueeze(-1))
        return torch.cat([left_mul * u, right_mul * v], dim = 1)
        
class EquivariantMLP(nn.Module):
    def __init__(self, n, hidden_dim, n_layers, n_input_layers, rank):
        super().__init__()
        self.r = rank
        self.n = n
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_input_layers = n_input_layers
        self.equivariant_layers = nn.ParameterList()
        self.first_equiv = nn.Parameter(torch.randn(2*n_input_layers, self.hidden_dim, n) / (n**.5))
        for i in range(1, n_layers):
            layer = nn.Parameter(torch.randn(2*n_input_layers, self.hidden_dim, self.hidden_dim) / (self.hidden_dim)**.5)
            self.equivariant_layers.append(layer)
        self.nonlins = nn.ModuleList([GLNonlinFast() for i in range(n_layers - 1)])
    def forward(self,uvs): #list of tuples (u,v), each tuple is shape b x n x r. There are l tuples
        """Forward method of EquivariantMLP. This method first applies a separate equivariant linear transform to each 
            Assumes that uvs is b x 2l x n x r, where in each batch, first all us are listed, then all vs are listed."""
        uvs = vmap_bmm_N0_new(self.first_equiv, uvs)
        #uvs = torch.bmm(self.first_equiv, uvs)
        #uvs = torch.matmul(self.first_equiv, uvs)
        for i in range(self.n_layers-1):
            uvs = self.nonlins[i](uvs)
            uvs = vmap_bmm_N0_new(self.equivariant_layers[i], uvs)
            
        return uvs

class GLInvariantMLP(nn.Module):
    def __init__(self, n, n_input_layers,rank,  out_dim, hidden_dim_equiv = 128, n_layers = 0, hidden_dim_inv = 128):
        super().__init__()
    
        self.equiv_mlp = EquivariantMLP(n, hidden_dim_equiv, n_layers, n_input_layers, rank)
        self.invariant_head = InvariantHead(hidden_dim_equiv, hidden_dim_inv, out_dim, n_input_layers)
    def forward(self, uvs):
        uvs_1 = self.equiv_mlp(uvs)
        uvs_2 = self.invariant_head(uvs_1)
        return uvs_2

class InvariantHead(nn.Module):
    def __init__(self, n, hidden_dim, out_dim, n_input_layers):
        super().__init__()
        self.n = n
        total_length = n ** 2 * n_input_layers
        self.linear1 = nn.Linear(total_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, uvs):

        u,v = torch.chunk(uvs, 2, dim = 1) #Both b x l x n x r
        prods = vmap_bmm_00(u,v.swapaxes(-1,-2)) #should be b x l x n x n
        out = prods.flatten(start_dim = 1) #b x l*n^2
        out = self.linear1(out)
        out = self.ln1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out
