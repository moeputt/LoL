import torch
from torch import nn
from torch.nn import functional as F
from scipy.linalg import orthogonal_procrustes
def apply_A_matmul(A, u, v):
    return (u @ A).detach(), (torch.linalg.inv(A) @ v.T).T.detach()
def apply_A_conv(A, u, v):
    return (
        F.conv_transpose2d(u, A.unsqueeze(2).unsqueeze(3)).detach(),
        F.conv_transpose2d(torch.linalg.inv(A).unsqueeze(2).unsqueeze(3), v).detach()
    )
def fixup(t, mode):
    if len(t.shape) == 2:
        return t.movedim(1,-1).flatten(end_dim = -2)
    else:
        if mode == 'u':
            return t.movedim(1, -1).flatten(end_dim = -2)
        else:
            return t.movedim(0, -1).flatten(end_dim = -2)
class OAligner():
    def __init__(self, base_u, base_v):
        self.conv = len(base_u.shape) > 2
        self.base_u = fixup(base_u, 'u')
        
        self.base_v = fixup(base_v, 'v')
        
        self.base = torch.cat([self.base_u, self.base_v], dim = 0)
        
        self.apply_fn = apply_A_conv if self.conv else apply_A_matmul
        
    def __call__(self, us, vs):
        out_us = []
        out_vs = []

        for i, (u, v) in enumerate(zip(us, vs)):
            u, v = u.cpu(), v.cpu()
            fixed_u = fixup(u, 'u')
            fixed_v = fixup(v, 'v')
            to_align = torch.cat([fixed_u, fixed_v], dim = 0).cpu()
            q = torch.tensor(orthogonal_procrustes(to_align, self.base)[0])
            out_u, out_v = self.apply_fn(q, u, v)
            out_us.append(out_u)
            out_vs.append(out_v)
        return torch.stack(out_us), torch.stack(out_vs)
            
            
class ModelAligner():
    def __init__(self, base_uvs):
        self.aligners = []
        
        for base_u, base_v in base_uvs:
            self.aligners.append(OAligner(base_u, base_v))
    def __call__(self, uvs, *args, **kwargs):
        out_uvs = []
        for i, (u, v) in enumerate(uvs):
            out_uvs.append(self.aligners[i](u, v, *args, **kwargs))
        return out_uvs