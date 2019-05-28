import torch
import torch.nn as nn
#from torch.jit import Tensor

#import torch.jit as jit
from torch import Tensor
from typing import List, Tuple


class TPRU(torch.jit.ScriptModule):


  def __init__(self, config):
    super(TPRU, self).__init__()

    self.basis = nn.Parameter(torch.randn(config.n_roles, config.d_hidden))
    self.basis.requires_grad = False

    self.v2ru = nn.Linear(config.d_hidden, config.d_hidden * 2)
    self.h2fg = nn.Linear(config.d_hidden, config.d_hidden * 2)
    self.x2fg = nn.Linear(config.d_input,  config.d_hidden * 2)

    self.bias_h = nn.Parameter(torch.ones(1))
    self.bias_x = nn.Parameter(torch.ones(1))

  @torch.jit.script_method
  def forward(self, inputs, masks, state):   
    h = state 
   
    x2f, x2g = self.x2fg(inputs).chunk(2, -1)
    v2r, v2u = self.v2ru(self.basis).chunk(2, -1)

    v2u = v2u.transpose(1,0)

    seql, bsize, dim = x2f.size()

    fxt = torch.relu(x2f.view(-1, dim).mm(v2u) + self.bias_x).view(seql, bsize, -1)

    outputs = torch.jit.annotate(List[Tensor], [])

    for i in range(inputs.size(0)):
      h2f, h2g = self.h2fg(h).chunk(2, -1)
      fbt = torch.relu(h2f.mm(v2u) + self.bias_h)
      fxb = (fxt[i] + fbt).pow(2.0)
      ft = fxb / fxb.sum(dim=1, keepdim=True).clamp_min(1e-7)
      h_ = ft.mm(v2r)

      gt = torch.sigmoid(x2g[i] + h2g)
      h = gt * h_ + (1. - gt) * h
      h = h * masks[i]
      
      outputs += [h]      
    
    return torch.stack(outputs)
