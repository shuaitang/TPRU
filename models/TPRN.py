import torch
import torch.nn as nn

#from torch.jit import Tensor
from torch import Tensor
from typing import List, Tuple

from TPRU import TPRU

class TPRULayer(torch.jit.ScriptModule):
#class TPRULayer(torch.nn.Module):

  __constants__ = ['n_layers', 'bidirectional', 'layers', 'lasth']
  def __init__(self, config):
    super(TPRULayer, self).__init__()

    config.d_input = config.d_embed
    lst = []

    for n in range(config.n_layers):         
      lst.append(TPRU(config))
      if config.bidirectional:
        lst.append(TPRU(config))
      config.d_input = config.d_hidden * (2 if config.bidirectional else 1)

    self.n_layers = config.n_layers
    self.bidirectional = config.bidirectional
    self.layers = nn.ModuleList(lst)
    self.lasth = config.lasth

  
  @torch.jit.script_method
  def forward(self, padded, masks, indices, state):
    hn = torch.jit.annotate(List[Tensor], [])    
    bsize = padded.size(1)
    ind_vecs = 0
    lens = masks.sum(dim=0)

    outputs = torch.jit.annotate(List[Tensor], [])

    if self.bidirectional:
      for mod in self.layers:
        c_inputs, c_masks = (padded, masks) if ind_vecs % 2 == 0 else (torch.flip(padded, [0]), torch.flip(masks, [0]))
        out = mod(c_inputs, c_masks, state[ind_vecs])
        out = out if ind_vecs % 2 == 0 else torch.flip(out, [0])
        outputs += out
        padded = padded if ind_vecs % 2 == 0 else torch.cat(outputs[-2:], dim=2)
        ind_vecs += 1
    else:
      for mod in self.layers:
        out = mod(padded, masks, state[ind_vecs])
        outputs += out
        padded = outputs[-1]
        ind_vecs += 1

    lasth = torch.masked_select(torch.stack(outputs), indices.byte().unsqueeze(0)).view(self.n_layers * (2 if self.bidirectional else 1), bsize, -1)
      
    return padded, lasth




