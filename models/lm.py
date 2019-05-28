import torch 
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import os

from TPRN import TPRULayer

class model(nn.Module):

  def __init__(self, config):
    super(model, self).__init__()
    self.config = config
    self.hidden_size = config.d_hidden 

    self.embed = nn.Embedding(config.n_embed, config.d_embed, padding_idx=0)
    self.en = TPRULayer(config)
    self.proj = nn.Linear(config.d_hidden * (2 if config.bidirectional else 1), config.d_embed)
    self.out = nn.Linear(config.d_embed, config.n_embed)

    self.h0 = nn.Parameter(torch.zeros(config.n_layers * (2 if config.bidirectional else 1), 1, config.d_hidden))
    self.h0.requires_grad = False

    if config.tie_weights:
      self.out.weight = self.embed.weight

    self.init_weights()


  def init_weights(self):
    torch.nn.init.orthogonal_(self.embed.weight.data)
    torch.nn.init.orthogonal_(self.proj.weight.data)
    self.proj.bias.data.fill_(0.)


  def loss_function(self, pred, label):
    return F.cross_entropy(pred, label, ignore_index=0)


  def encoding(self, inputs, masks, state=None):

    inputs = inputs.transpose(0,1)
    masks = masks.transpose(0,1)

    with torch.no_grad():
      shifted_masks = torch.zeros_like(masks)
      shifted_masks[:-1] = masks[1:]
      indices = (masks - shifted_masks).unsqueeze(0).byte()

    outputs, hn = self.en(inputs, masks, indices, state if state is not None else self.h0)
    outputs = outputs.transpose(0,1)

    return outputs, hn


  def forward(self, tokens, h0=None):

    # Encoding
    wembs = self.embed(tokens[:, :self.config.seq_length]) 
    masks = torch.ones_like(wembs)[:,:,0:1]
    output, hn = self.encoding(wembs, masks, h0)
    pred = self.proj(output)
    pred = self.out(pred)

    loss = self.loss_function(pred.view(-1, self.config.n_embed), tokens[:, 1:].contiguous().view(-1))

    return loss.view(1)

