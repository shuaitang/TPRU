import torch
import torch.nn as nn

#from torch.jit import Tensor
from torch import Tensor
from typing import List, Tuple

from TPRU import TPRU

#class TPRULayer(torch.jit.ScriptModule):
class TPRULayer(torch.nn.Module):

#  __constants__ = ['n_layers', 'bidirectional', 'f_layers', 'b_layers']
  def __init__(self, config):
    super(TPRULayer, self).__init__()

    config.d_input = config.d_embed

    f_lst = []
    b_lst = []

#    f_basis = []
#    b_basis = []

    for n in range(config.n_layers):         
      f_lst.append(TPRU(config))
#      basis = nn.Parameter(torch.randn(config.n_roles, config.d_hidden))
#      basis.requires_grad = False
#      f_basis.append(basis)
      if config.bidirectional:
        b_lst.append(TPRU(config))
#        basis = nn.Parameter(torch.randn(config.n_roles, config.d_hidden))
#        basis.requires_grad = False
#        b_basis.append(basis)
      config.d_input = config.d_hidden * (2 if config.bidirectional else 1)

    self.n_layers = config.n_layers
    self.bidirectional = config.bidirectional
    self.f_layers = nn.ModuleList(f_lst)
    self.b_layers = nn.ModuleList(b_lst)
#    self.f_basis = nn.ParameterList(f_basis)
#    self.b_basis = nn.ParameterList(b_basis)

    self.h0 = nn.Parameter(torch.zeros(1, config.d_hidden))

   


#  @torch.jit.script_method
  def forward(self, padded, masks, state=None):
#    hn = torch.jit.annotate(List[Tensor], [])    
    hn = []
    #state: num_layers * num_directions, batch, hidden_size
    bsize = padded.size(1)
    ind_vecs = 0
    lens = masks.sum(dim=0)

    

    if self.bidirectional:
      for f_mod, b_mod in zip(self.f_layers, self.b_layers):
        f_output = f_mod(padded, masks, state[ind_vecs] if state is not None else self.h0)        
        ind_vecs += 1
        b_output = torch.flip(b_mod(torch.flip(padded, [0]), torch.flip(masks, [0]), state[ind_vecs] if state is not None else self.h0), [0])
        ind_vecs += 1
        for i in range(bsize):
          hn += [f_output[lens[i] - 1]]
          hn += [b_output[lens[i] - 1]]
        output = torch.cat([f_output, b_output], dim=2)
        padded = output

    else:
      for f_mod in self.f_layers:
        output = f_mod(padded, masks, state[ind_vecs] if state is not None else self.h0)
        for i in range(bsize):
          hn += [output[lens[i] - 1]]
        ind_vecs += 1
        padded = output

    return output, torch.stack(hn)

