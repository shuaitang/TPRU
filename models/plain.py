import torch 
import torch.nn as nn
import numpy as np

from torch.nn import functional as F

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
    self.clser = classifier(self.config)    
    self.h0 = nn.Parameter(torch.zeros(config.n_layers * (2 if config.bidirectional else 1), 1, config.d_hidden))
    self.h0.requires_grad = False

  def init_weights(self):
    torch.nn.init.orthogonal_(self.embed.weight.data)


  def rep_pooling(self, seqs, lasth=None, lens=None):
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


  def encoding(self, inputs, masks, h0):

    #inputs: NxLxC
    inputs = inputs.transpose(0,1)
    masks = masks.transpose(0,1).unsqueeze(2)
    #x: LxNxC

    with torch.no_grad():
      shifted_masks = torch.zeros_like(masks)
      shifted_masks[:-1] = masks[1:]
      indices = (masks - shifted_masks).unsqueeze(0).byte()
 
    outputs, _ = self.en(inputs, masks, indices, h0)
    outputs = outputs.transpose(0,1)

    z = self.rep_pooling(outputs)
    

    return z


  def forward(self, t_pre, t_hypo, mask_pre=None, mask_hypo=None):

    # Encoding
    if self.config.elmo:
      tokens = t_pre + t_hypo
      wemb, masks = self.embed(tokens)
      wemb = torch.cat([wemb[0], wemb[1]], dim=2)

    else:
      wemb = self.embed(torch.cat([t_pre, t_hypo], dim=0))
      masks = torch.cat([mask_pre, mask_hypo], dim=0)
     
    u, v = self.encoding(wemb, masks, self.h0).chunk(2,0)

    # Classification
    pred = self.clser(u, v)

    return pred   
