import math
import torch
from torch import nn
from torch.nn.init import kaiming_normal_
from torch.nn import functional as F

vmap_linear = torch.vmap(F.linear)
vmap_bmm_N0 = torch.vmap(torch.bmm, in_dims = (None, 0))
vmap_bmm_00 = torch.vmap(torch.bmm, in_dims = (0, 0))
vmap_linear_10 = torch.vmap(F.linear, in_dims = (1,0))
device = torch.device('cuda')

class EquivDropout(nn.Module):
    def __init__(self, p = .5):
        super().__init__()
        self.p = p
    def forward(self, u, v):
        b,l,n,r = u.shape
        if self.training:
            left = (torch.rand(b,l,n,1) > self.p).to(device)
            right = (torch.rand(b,l,n,1) > self.p).to(device)
            return left * u, right * v
        else:
            return u * (1-self.p), v * (1-self.p)
        
class GLNorm(nn.Module):
    def __init__(self, n, l):
        super().__init__()
        self.scale_u = nn.Parameter(torch.ones(1, l, n, 1))
        self.scale_v = nn.Parameter(torch.ones(1, l, n, 1))
        
    def forward(self, u,v, eps = 1e-6): #bxlxnxr, bxlxnx
        b,l,n,r = u.shape
        uu = vmap_bmm_00(u.swapaxes(-1,-2), u)
        vv = vmap_bmm_00(v.swapaxes(-1,-2), v)
        uuvv = vmap_bmm_00(uu, vv).diagonal(dim1 = -2, dim2 = -1)
        norms = (uuvv.sum(dim = -1) ** .5).reshape(b,l,1,1) + eps
        # return u,v
        return self.scale_u * u / (norms ** .5), self.scale_v * v / (norms **.5)
class GLNorm2(nn.Module):
    def forward(self, u,v,eps=1e-6):
        to_norm = vmap_bmm_00(u, v.swapaxes(-1,-2))
        left_mul = F.relu(torch.sign((to_norm.sum(dim=3))).unsqueeze(-1))
        right_mul = F.relu(torch.sign((to_norm.sum(dim=2))).unsqueeze(-1))
        return left_mul * u, right_mul * v
        
class EquivariantLinear(nn.Module):
    def __init__(self, n, m, out_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_dim, n+m)/(n+m)**.5)
    def forward(self, u, v):
        uv = torch.cat([u,v], dim = 1)
        return self.w @ uv
class EquivariantLinear2(nn.Module):
    def __init__(self, n, m, out_dim):
        super().__init__()
        self.w_1 = nn.Parameter(torch.randn(out_dim // 2, n)/(n)**.5)
        self.w_2 = nn.Parameter(torch.randn(out_dim // 2, m)/(m)**.5)
        
    def forward(self, u, v):
        uv = torch.cat([self.w_1 @ u, self.w_2 @v], dim = 1)
        return uv
class GLNonLinWide(nn.Module):
    def __init__(self, n, l, k=4):
        super().__init__() # b x l x 2n x r
        self.n = n
        self.ws_1 = nn.Parameter(torch.randn(l, k, n, device = device) / (n)**.5)
        self.ws_2 = nn.Parameter(torch.randn(l, k, n, device = device) / (n)**.5)
        self.ln1 = nn.GroupNorm(1, l)
        self.lnR1 = nn.GroupNorm(1, l)
        self.lnL1 = nn.GroupNorm(1, l)
        self.lnR2 = nn.GroupNorm(1, l)
        self.lnL2 = nn.GroupNorm(1, l)
        self.linear_10 = nn.Linear(l*k**2, l*n*k, device = device)
        self.linear_20 = nn.Linear(l*k**2, l*n*k, device = device)
        self.linear_11 = nn.Linear(l*n*k, l*n**2, device = device)
        self.linear_21 = nn.Linear(l*n*k, l*n**2, device = device)
        
    def forward(self, u, v):
        l = len(u[0])
        new_ws_1 = vmap_bmm_N0(self.ws_1, u)
        new_ws_2 = vmap_bmm_N0(self.ws_2, v)
        
        linear_outs = vmap_bmm_00(new_ws_1, new_ws_2.swapaxes(-1,-2))
        linear_outs = linear_outs.flatten(start_dim=1)
        left_mul = self.linear_10(linear_outs) 
        left_mul = F.relu(left_mul)
        left_mul = self.linear_11(left_mul)

        
        right_mul = self.linear_20(linear_outs)
        right_mul = F.relu(right_mul)
        right_mul = self.linear_21(right_mul)

        left_mul = left_mul.reshape(left_mul.shape[0], l, self.n, self.n) 
        
        left_mul = (left_mul) / (self.n**.5)
        
        
        right_mul = right_mul.reshape(right_mul.shape[0], l, self.n, self.n)
        right_mul = (right_mul) / (self.n**.5)
        #print(1, u.std().item())
        u = vmap_bmm_00(left_mul, u)
        #print(2, u.std().item())
        
        v = vmap_bmm_00(right_mul, v)
        
        return u,v
class GLNonLin(nn.Module):
    def __init__(self, n, l, k=4):
        super().__init__() # b x l x 2n x r
        self.n = n
        self.ws_1 = nn.Parameter(torch.randn(l, k, n, device = device) / (n)**.5)
        self.ws_2 = nn.Parameter(torch.randn(l, k, n, device = device) / (n)**.5)
        self.ln1 = nn.GroupNorm(1, l)
        self.lnR1 = nn.GroupNorm(1, l)
        self.lnL1 = nn.GroupNorm(1, l)
        self.lnR2 = nn.GroupNorm(1, l)
        self.lnL2 = nn.GroupNorm(1, l)
        self.MLP_10 = nn.Parameter(torch.randn(l, n*k, k**2, device = device) / k)
        self.biases_10 = nn.Parameter(torch.randn(l, 1, n*k, device = device)/(k))
        self.MLP_20 = nn.Parameter(torch.randn(l, n*k, k**2, device = device) / k)
        self.biases_20 = nn.Parameter(torch.zeros(l, 1, n*k, device = device)/(k))
        
        self.MLP_11 = nn.Parameter(torch.randn(l, n**2, k*n, device = device) / (n*k)**.5)
        self.biases_11 = nn.Parameter(torch.zeros(l, 1, n**2, device = device)/(n*k)**.5)
        
        self.MLP_21 = nn.Parameter(torch.randn(l, n**2, k*n, device = device) / (n*k)**.5)
        self.biases_21 = nn.Parameter(torch.zeros(l, 1, n**2, device = device)/(n*k)**.5)
        
        
        
        
    def forward(self, u, v):
        l = len(u[0])
        new_ws_1 = vmap_bmm_N0(self.ws_1, u)
        new_ws_2 = vmap_bmm_N0(self.ws_2, v)
        
        linear_outs = vmap_bmm_00(new_ws_1, new_ws_2.swapaxes(-1,-2))
        linear_outs = linear_outs.flatten(start_dim=2).swapaxes(0,1)

        left_mul = vmap_linear(linear_outs, self.MLP_10)     
        left_mul = left_mul + self.biases_10
        left_mul = F.relu(left_mul)
        left_mul = vmap_linear(left_mul, self.MLP_11)
        left_mul = (left_mul + self.biases_11).swapaxes(0,1)

        
        right_mul = vmap_linear(linear_outs, self.MLP_20)
        right_mul = right_mul + self.biases_20
        right_mul = F.relu(right_mul)
        right_mul = vmap_linear(right_mul, self.MLP_21)
        right_mul = (right_mul + self.biases_21).swapaxes(0,1)

        
        left_mul = left_mul.reshape(left_mul.shape[0], l, self.n, self.n) 
        left_mul = (left_mul) / (self.n**.5)
        
        right_mul = right_mul.reshape(right_mul.shape[0], l, self.n, self.n)
        right_mul = (right_mul) / (self.n**.5)
        
        u = vmap_bmm_00(left_mul, u)
        v = vmap_bmm_00(right_mul, v)
        
        return u,v
        
        
        
class EquivariantMLP(nn.Module):
    def __init__(self, n, m, hidden_dim, n_layers, n_input_layers, rank, mode = 'O', k=4, p =.5):
        super().__init__()
        self.r = rank
        self.n = n
        self.m = m
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_input_layers = n_input_layers
        self.equivariant_layers = nn.ModuleList()
        self.equivariant_layers_u = nn.ParameterList()
        self.equivariant_layers_v = nn.ParameterList()
        self.norms = nn.ModuleList([GLNorm(hidden_dim, n_input_layers) for i in range(n_layers - 1)])
  
        self.first_layer = nn.ModuleList([EquivariantLinear2(n, m, 2*hidden_dim) for n,m in zip(n, m)])
        for i in range(1, n_layers):
            layer_u = nn.Parameter(torch.randn(n_input_layers, self.hidden_dim, self.hidden_dim) / (self.hidden_dim)**.5)
            layer_v = nn.Parameter(torch.randn(n_input_layers, self.hidden_dim, self.hidden_dim) / (self.hidden_dim)**.5)
            
            self.equivariant_layers_u.append(layer_u)
            self.equivariant_layers_v.append(layer_v)
            
        self.nonlins = nn.ModuleList([GLNorm2() for i in range(n_layers - 1)])
        self.dropouts = nn.ModuleList([EquivDropout(p) for i in range(n_layers - 1)])

    def forward(self, uvs): #list of tuples (u,v), each tuple is shape b x n x r. There are l tuples
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
        for i in range(self.n_layers):
            #-----------Attention + Norm + Skip + MLP------------#
            if (i != self.n_layers - 1):
                # us_1, vs_1 = self.norms[i](us, vs)
                us_1, vs_1 = self.nonlins[i](us, vs)
                us_1, vs_1 = self.dropouts[i](us_1, vs_1)
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
    def __init__(self,ns, ms, n_input_layers,rank,  out_dim,hidden_dim_equiv = 128, n_layers = 0, hidden_dim_inv = 128, d = 16, layer_types = None, w = 2, k=4, p=.5, pool_mean=False, head_type=1):
        super().__init__()
    
        self.equiv_mlp = EquivariantMLP(ns, ms, hidden_dim_equiv, n_layers, n_input_layers, rank, mode = 'GL', k=k,p=p)
        new_ns, new_ms = [hidden_dim_equiv for i in range(n_input_layers)], [hidden_dim_equiv for i in range(n_input_layers)]
        if head_type == 1:
            self.invariant_head = InvariantHead(new_ns, new_ms, hidden_dim_inv, out_dim, n_input_layers, d = d, layer_types = layer_types, w=w, pool_mean=pool_mean, p=p)
        elif head_type == 2:
            self.invariant_head = InvariantHead2(new_ns, new_ms, hidden_dim_inv, out_dim, n_input_layers, d = d, layer_types = layer_types, w=w, pool_mean=pool_mean, p=p)
            

    def forward(self, uvs):
        uvs_1 = self.equiv_mlp(uvs)
        uvs_2 = self.invariant_head(uvs_1)
        
        return uvs_2

class InvariantHead(nn.Module):
    def __init__(self, ns, ms, hidden_dim, out_dim, n_input_layers, d = 16, w = 2, layer_types = None, pool_mean=False, p=0.0):
        if layer_types is None:
            self.layer_types = [0 for i in (ns)]
        else:
            self.layer_types = layer_types
        super().__init__()
        self.ns = ns
        self.ms = ms
        self.invariant_outputs = nn.ModuleList([InvariantOutput(n, m, d, layer_type, width_mul = w, pool_mean=pool_mean) for n, m, layer_type in zip(ns, ms, self.layer_types)])
        
        total_length = sum([output.num_elem for output in self.invariant_outputs])
        self.linear1 = nn.Linear(total_length, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(p=p)

    def forward(self, uvs):

        out = torch.cat([f(u,v).flatten(start_dim=1) for (u,v), f in zip(uvs, self.invariant_outputs)],dim=1)
        out = self.dropout(out)
        
        out = self.linear1(out)
        out = self.ln1(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out

class InvariantHead2(nn.Module):
    def __init__(self, ns, ms, hidden_dim, out_dim, n_input_layers, d = 16, w = 2, layer_types = None, pool_mean=False, p=0.0):
        if layer_types is None:
            self.layer_types = [0 for i in (ns)]
        else:
            self.layer_types = layer_types
        super().__init__()
        self.ns = ns
        self.ms = ms
        assert not pool_mean
        self.invariant_outputs = nn.ModuleList([InvariantOutput(n, m, d, layer_type, width_mul = w, pool_mean=pool_mean) for n, m, layer_type in zip(ns, ms, self.layer_types)])
        
        # requires all to be same size, hidden_dim x hidden_dim
        self.dropout = nn.Dropout(p=p)
        self.linear1 = nn.Linear(self.invariant_outputs[0].num_elem, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim*n_input_layers, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, uvs):

        out = torch.stack([f(u,v).flatten(start_dim=1) for (u,v), f in zip(uvs, self.invariant_outputs)], dim=1) # b x l x d
        out = self.dropout(out)
        
        out = self.linear1(out)
        out = self.ln1(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = out.flatten(start_dim=1)
        out = self.linear2(out)
        out = self.ln2(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out


class SimpleNet(nn.Module):
    def __init__(self, ns, ms, hidden_dim, out_dim):
        super().__init__()
        self.ns = ns
        self.ms = ms
        self.lin_1 = nn.Linear((sum(ns) + sum(ms))*4, hidden_dim)
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
    def __init__(self, n,m, d = 16, layer_type = 0, width_mul = 2, pool_mean=False):
        super().__init__()
        self.layer_type = layer_type
        self.d = d
        self.pool_mean = pool_mean
        if self.layer_type == 0:
            self.m = m
            self.n = n
            self.x = nn.Parameter(torch.randn(d, m))
            self.num_elem = 1 if pool_mean else m*n
        else:
            self.kernel_size, self.stride, self.padding = layer_type, 1, 0
            width = round(width_mul * (self.kernel_size))
            tot = (self.kernel_size ** 2 + 1)
            num_in_channels = (n+m)//tot
            num_out_channels = (n+m) - (self.kernel_size ** 2) * num_in_channels
            self.m = num_in_channels
            self.n = num_out_channels
            self.out_image_dim = (width + 2 * self.padding - (self.kernel_size) ) // self.stride + 1
            self.num_elem = self.d * self.n  * (self.out_image_dim)**2
            
            self.x = nn.Parameter(torch.randn(d, self.m, width, width))

    def forward(self, u, v):
        if self.layer_type == 0:
            bs = u.shape[0]
            if self.pool_mean:
                return torch.bmm(u, v.swapaxes(-1,-2)).mean(dim = [1,2]).unsqueeze(-1)
            else:
                return torch.bmm(u, v.swapaxes(-1,-2)).reshape(bs, -1)
        else:
            uv = torch.cat([u,v], dim = 1)
            b = len(uv)
            w = self.x.shape[-1]
            first_conv = uv[:, -self.m * self.kernel_size ** 2:]
            second_conv = uv[:, :-self.m * self.kernel_size ** 2]
            
            r = uv.shape[-1]
            first_conv = first_conv.reshape(b, self.m, self.kernel_size, self.kernel_size, r).permute(0, 4, 1, 2, 3)
            second_conv = second_conv.reshape(b, self.n, r, 1, 1)
            
            new_xs = self.x.repeat([1, b, 1, 1])
            new_first_conv = first_conv.flatten(end_dim = 1)
            new_second_conv = second_conv.flatten(end_dim = 1)
            first_out = F.conv2d(new_xs, new_first_conv, groups = b, stride = self.stride, padding = self.padding)
            second_out = F.conv2d(first_out, new_second_conv, groups = b).reshape(self.d, b, self.n, self.out_image_dim,self.out_image_dim).swapaxes(0,1)
            return second_out


        
class TransformerHead(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, seqlen, p=.1):
        # n x m matrix
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.encoder1 = nn.TransformerEncoderLayer(hidden_dim, nhead=hidden_dim//16, dim_feedforward=hidden_dim*2, batch_first=True, dropout=p, norm_first=True, bias=False)
        self.encoder2 = nn.TransformerEncoderLayer(hidden_dim, nhead=hidden_dim//16, dim_feedforward=hidden_dim*2, batch_first=True, dropout=p, norm_first=True, bias=False)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) / math.sqrt(hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, seqlen+1, hidden_dim) / math.sqrt(hidden_dim))
        self.out_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(p), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        # x is b x l x d
        x = self.in_proj(x)
        b, l, d = x.shape
        x = torch.cat([self.cls_token.repeat(b, 1, 1), x], dim=1)
        pe = self.pos_embed.repeat(b, 1, 1)
        x = x + pe
        x = self.encoder1(x)
        x = self.encoder2(x)
        out = x[:, 0] # get cls token embedding
        out = self.out_head(out)
        return out

class TempGLMLP(nn.Module):
    def __init__(self, n, m, hidden_dim, out_dim, p=.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.first_layer = nn.ModuleList([EquivariantLinear2(n, m, 2*hidden_dim) for n,m in zip(n, m)])
        seqlen = len(n)
        self.norm = nn.LayerNorm(hidden_dim**2)
        self.inv_head = TransformerHead(hidden_dim**2, hidden_dim, out_dim, seqlen, p=p)

    def forward(self, uvs):
        # uvs is list of tuples (u,v), where u @ v.T is a lora delta for a layer
        us = []
        vs = []
        for j, (u,v) in enumerate(uvs):
            out = self.first_layer[j](u,v)
            # now each u and v is hidden_dim x r
            u, v = torch.split(out, self.hidden_dim, dim = 1)
            us.append(u)
            vs.append(v)
        # b x num_layers x hidden_dim x r
        us = torch.stack(us, dim=1)
        vs = torch.stack(vs, dim=1)
        bs, l, d, r = us.shape
        inv_mult = torch.bmm(us.reshape(bs*l, d, r), vs.reshape(bs*l, d, r).transpose(-1, -2))
        # bs*l x d x d
        inv_mult = inv_mult.reshape(bs, l, d, d)
        # bs x l x d^2
        inv_mult = inv_mult.reshape(bs, l, -1)
        inv_mult = self.norm(inv_mult)
        out = self.inv_head(inv_mult)
        return out
