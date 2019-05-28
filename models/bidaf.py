import torch 
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from TPRN import TPRULayer
from classifier import classifier
from elmo import ELMoEmbedding


class model(nn.Module):

  def __init__(self, config):
    super(model, self).__init__()
    self.config = config
    self.hidden_size = config.d_hidden 

    self.embed = ELMoEmbedding(config) if config.elmo else nn.Embedding(config.n_embed, config.d_embed, padding_idx=0)
    self.en = TPRULayer(config)
    self.clser = classifier(config) 
    self.en_h0 = nn.Parameter(torch.zeros(config.n_layers * (2 if config.bidirectional else 1), 1, config.d_hidden))
    self.en_h0.requires_grad = False


    att_config = config
    att_config.d_embed = config.d_embed + config.d_hidden * 2 * (2 if config.bidirectional else 1)
    att_config.n_layers = config.att_layers
    self.att = TPRULayer(att_config)
    self.att_h0 = nn.Parameter(torch.zeros(config.att_layers * (2 if config.bidirectional else 1), 1, config.d_hidden))
    self.att_h0.requires_grad = False



  def init_weights(self):
    torch.nn.init.orthogonal_(self.embed.weight.data) 


  def rep_pooling(self, seqs=None, lasth=None, lens=None):
    #seqs: NxLxC
    if self.config.pooling == "max":
      seqs.data.masked_fill_((seqs==0).byte().data, -float('inf'))
      return torch.max(seqs, dim=1)[0]
    elif self.config.pooling == "min":
      seqs.data.masked_fill_((seqs==0).byte().data, float('inf'))
      return torch.min(seqs, dim=1)[0]
    elif self.config.pooling == "avg":
      return torch.sum(seqs, dim=1) / lens
    else:
      return lasth


  def encoding(self, en, inputs, masks, h0):

    #x: NxLxC
    inputs = inputs.transpose(0,1)
    masks = masks.transpose(0,1).unsqueeze(2)
    #x: LxNxC
    outputs = en(inputs, masks, h0)
    outputs = outputs.transpose(0,1) #NxLxC

    return outputs


  def attn(self, u, v):
    # u, v: NxLxC
    H = u.bmm(v.transpose(1,2)) # NxLxL
    Au = F.softmax(H, dim=1)
    Av = F.softmax(H, dim=2)
    v_ = Av.bmm(v)
    u_ = Au.transpose(1,2).bmm(u)
    return u_, v_ 


  def forward(self, t_pre, t_hypo, mask_pre=None, mask_hypo=None):

    # Encoding
    if self.config.elmo:
      tokens = t_pre + t_hypo
      wemb, masks = self.embed(tokens)
    else:
      wemb = self.embed(torch.cat([t_pre, t_hypo], dim=0))
      masks = torch.cat([mask_pre, mask_hypo], dim=0)

    u, v = self.encoding(self.en, wemb[0] if self.config.elmo else wemb, masks, self.en_h0).chunk(2,0)
    u_, v_ = self.attn(u, v)

    U = torch.cat([u, v_], dim=2)
    V = torch.cat([v, u_], dim=2)
    attn_input = torch.cat([U,V], dim=0)
    attn_input = torch.cat([attn_input, wemb[1] if self.config.elmo else wemb], dim=2)

    z = self.encoding(self.att, attn_input, masks, self.att_h0)
    z = self.rep_pooling(z)
    z_pre, z_hypo = z.chunk(2,0)

    # Classification
    pred = self.clser(z_pre, z_hypo)

    return pred
    
