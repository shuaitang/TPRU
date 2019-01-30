import torch
import torch.nn as nn
from torch.nn import functional as F

class TPRU(nn.Module):

  def __init__(self, config, d_input):
    super(TPRU, self).__init__()
    self.config = config

    self.h2fg = nn.Linear(config.d_hidden, config.d_hidden * 2)

    self.bh = nn.Parameter(torch.ones(1).float())
    self.bx = nn.Parameter(torch.ones(1).float())
    self.bg = nn.Parameter(torch.ones(1).float())     

#    self.init_weights()


#  def init_weights(self):
#    for m in self.modules():
#      if isinstance(m, nn.Linear):
#        torch.nn.init.xavier_normal_(m.weight.data)
#        if m.bias is not None:
#          m.bias.data.fill_(0.)


  def forward(self, h, x, v2ru):

    h = h.clone()
    bs, d_h = h.size()

    x2f, x2g = x.chunk(2,1)
    h2f, h2g = self.h2fg(h).chunk(2,1)

    vr, vu = v2ru.chunk(2,-1)

    x2f = x2f.view(bs, 1, self.config.n_slices, -1)
    h2f = h2f.view(bs, 1, self.config.n_slices, -1)

    ah, ax = (torch.cat([x2f, h2f], dim=0) * vu).sum(dim=3).chunk(2,0)

    ah = torch.relu(ah + self.bh)
    ax = torch.relu(ax + self.bx)

    a = F.normalize((ah + ax).pow(2.), p=1., dim=1)

    hh = (a.unsqueeze(3) * vr).sum(dim=1).view(bs, -1)
          
    g = torch.sigmoid(x2g + h2g + self.bg)
    h_ = g * hh + (1-g) * h

    return h_
