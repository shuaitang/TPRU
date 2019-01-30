import torch 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from TPRU import TPRU


class TPRRNN(nn.Module):

  def __init__(self, config):
    super(TPRRNN, self).__init__()
    self.config = config
    
    self.f_cells = nn.ModuleList()
    self.f_x2fgs = nn.ModuleList()
    self.f_basis = nn.ParameterList()
    self.f_v2ru  = nn.ModuleList()

    if config.birnn:
      self.b_cells = nn.ModuleList()
      self.b_x2fgs = nn.ModuleList()
      self.b_basis = nn.ParameterList()
      self.b_v2ru = nn.ModuleList()   

    d_input = config.d_embed

    for n in range(config.n_layers):
      self.f_x2fgs.append(nn.Linear(d_input, config.d_hidden * 2))
      self.f_cells.append(TPRU(config, d_input))   
      self.f_basis.append(nn.Parameter(torch.randn(config.n_roles, config.d_hidden)))
      self.f_basis[n].requires_grad = False
      self.f_v2ru.append(nn.Linear(config.d_hidden, config.d_hidden * 2)) 

      if config.birnn:
        self.b_x2fgs.append(nn.Linear(d_input, config.d_hidden * 2))
        self.b_cells.append(TPRU(config, d_input))
        self.b_basis.append(nn.Parameter(torch.randn(config.n_roles, config.d_hidden)))
        self.b_basis[n].requires_grad = False
        self.b_v2ru.append(nn.Linear(config.d_hidden, config.d_hidden * 2))

      d_input = config.d_hidden * (2 if config.birnn  else 1)

#    self.init_weights()


#  def init_weights(self):
#    for m in self.modules():
#      if isinstance(m, nn.ModuleList):
#        for l in m:
#          if isinstance(l, nn.Linear):
#            torch.nn.init.xavier_normal_(l.weight.data)
#            if l.bias is not None:
#              l.bias.data.fill_(0.)
#


  def forward_rnn(self, en, batch_sizes, x2fg, v2ru, h0=None):

    bs = batch_sizes + [0]
    p = 0 
    q = bs[0] 
    h0 = h0 if h0 is not None else torch.zeros(bs[0], self.config.d_hidden)
    output = Variable(torch.zeros(sum(bs), self.config.d_hidden))
    hn = Variable(torch.zeros(bs[0], self.config.d_hidden))

    (h0, hn, output) = (h0.cuda(), hn.cuda(), output.cuda()) if self.config.cuda else (h0, hn, output)

    for i in range(len(bs) - 1):
      if i == 0:
        input_h = h0
      else:
        input_h = output[p:p+bs[i]]     
      p = q - bs[i]
      h = en(input_h, x2fg[p:q], v2ru)
      output[p:q] = h

      if bs[i] != bs[i+1]:
        hn[bs[i+1]:bs[i]] = h[bs[i+1]:bs[i]]

      q = q + bs[i+1]

    return output, hn



  def backward_rnn(self, en, batch_sizes, x2fg, v2ru, h0=None):
    bs = batch_sizes 
    q = np.sum(bs)
    p = q - bs[-1]
    h0 = h0 if h0 is not None else torch.zeros(bs[0], self.config.d_hidden)
    output = Variable(torch.zeros(sum(bs), self.config.d_hidden))
    hn = Variable(torch.zeros(bs[0], self.config.d_hidden))
 
    (h0, hn, output) = (h0.cuda(), hn.cuda(), output.cuda()) if self.config.cuda else (h0, hn, output)

    for i in range(len(bs)-1,-1,-1):
   
      if i == len(bs) - 1:
        input_h = h0[:bs[-1]]
      else:
        if bs[i] != bs[i+1]:
          input_h = torch.cat([output[p:q],h0[:bs[i]-bs[i+1]]], dim=0)
        else:
          input_h = output[p:q]
        q = p
        p = q - bs[i]


      h = en(input_h, x2fg[p:q], v2ru)
      output[p:q] = h
     
      if i == 0:
        hn = h 

    return output, hn



  def forward(self, x, h0=None):
    cnt = 0
    stride = self.config.n_layers * (2 if self.config.birnn else 1)
  
    if type(x) is torch.Tensor:
      output = x
      batch_sizes = [x.size(1)] * self.config.seq_length
    else:
      output = x.data
      batch_sizes = x.batch_sizes.cpu().detach().numpy().tolist()

    hn = Variable(torch.zeros(stride, batch_sizes[0], self.config.d_hidden))
    hn = hn.cuda() if self.config.cuda else hn

    for n in range(self.config.n_layers):

      x2fg = self.f_x2fgs[n](output)
      v2ru = self.f_v2ru[n](self.f_basis[n]).view(1, self.config.n_roles, self.config.n_heads, -1)
      f_output, f_hn = self.forward_rnn(self.f_cells[n], batch_sizes, x2fg, v2ru, h0[cnt] if h0 is not None else None)
      hn[cnt] = f_hn
      cnt += 1
      if self.config.birnn:
        x2fg = self.b_x2fgs[n](output)
        v2ru = self.b_v2ru[n](self.b_basis[n]).view(1, self.config.n_roles, self.config.n_heads, -1)
        b_output, b_hn = self.backward_rnn(self.b_cells[n], batch_sizes, x2fg, v2ru, h0[cnt] if h0 is not None else None)
        hn[cnt] = b_hn
        cnt += 1
      if self.config.birnn:
        output = torch.cat([f_output, b_output], dim=1)
      else:
        output = f_output

    output = torch.nn.utils.rnn.PackedSequence(output, x.batch_sizes)

    return output, hn



