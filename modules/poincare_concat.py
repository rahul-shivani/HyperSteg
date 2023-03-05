import torch
import torch.nn as nn
from scipy.special import beta

def sabs(x, eps: float = 1e-15):
    return x.abs().clamp_min(eps)

def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)


def arsinh(x: torch.Tensor):
    return (x + torch.sqrt(1 + x.pow(2))).clamp_min(1e-15).log().to(x.dtype)

def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2. * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    return 2 * z_norm / rc * arsinh(
        (2. * torch.matmul(rcx, z_unit) * drcr.cosh() - (1. + cx2) * drcr.sinh()) 
        / torch.clamp_min(1. - cx2, 1e-15))
    
def _project(x, k, dim: int = -1, eps: float = -1.0):
    if eps < 0:
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
    maxnorm = (1 - eps) / (sabs(k) ** 0.5)
    maxnorm = torch.where(k.lt(0), maxnorm, k.new_full((), 1e15))
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

class PoincareConcatLinear(nn.Module):
    def __init__(self, in_stacks, in_dim, out_dim, out_split=1, bias=True, ball=None, gain=1.):
        super().__init__()
        gain = 1. ###
        self.out_split = out_split
        self.ball = ball
        self.in_stacks = in_stacks
        self.in_dim = in_dim
        self.out_dim = out_dim
        weight = torch.empty(in_stacks * in_dim, out_dim).normal_( 
            mean=0, std=1. / (2 * self.in_dim * in_stacks * self.out_dim) ** 0.5 * gain)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.weight_v = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.empty(out_dim), requires_grad=bias)
        self.reset_parameters()
        self.beta_ni = beta(self.in_dim / 2, 1 / 2)
        self.beta_n = beta(self.in_dim * self.in_stacks / 2, 1 / 2)
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        size = x.size()
        x = self.ball.expmap0(x)
        x = self.ball.logmap0(x).contiguous().view(*size[:-2], self.in_stacks * self.in_dim)
        x = self.ball.expmap0(x * self.beta_n / self.beta_ni)
        return poincare_linear(
            x, 
            self.weight_g, 
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
            self.bias, 
            self.ball.c,
            self.out_split)
    
    def extra_repr(self):
        return (f'in_stacks={self.in_stacks},'
        f' in_dim={self.in_dim}, out_dim={self.out_dim}, bias={self.bias.requires_grad}')



def poincare_linear(x, weight_g, weight_v, bias, c, out_split : int = 1):
    rc = c.sqrt()
    x = unidirectional_poincare_mlr(x, weight_g, weight_v, bias, c)
    x = (rc * x).sinh() / rc
    if out_split > 1:
        size = x.size()
        x = x.view(*size[:-1], out_split, size[-1] // out_split)

    return _project(x / (1 + (1 + c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt()), -c, dim=-1)