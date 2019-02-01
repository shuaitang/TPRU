import torch 
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import os

from RNN import TPRU, TPRRNN

class model(nn.Module):

  def __init__(self, config):
    super(model, self).__init__()
    self.config = config
    self.hidden_size = config.d_hidden 

    self.embed = nn.Embedding(config.n_embed, config.d_embed, padding_idx=0)
    self.en = TPRRNN(config)
    self.proj = nn.Linear(config.d_hidden * (2 if config.birnn else 1), config.d_embed)
    self.out = nn.Linear(config.d_embed, config.n_embed)

    if config.tie_weights:
      self.out.weight = self.embed.weight

    self.init_weights()


  def init_weights(self):
    torch.nn.init.orthogonal_(self.embed.weight.data)
    torch.nn.init.orthogonal_(self.proj.weight.data)
    self.proj.bias.data.fill_(0.)


  def loss_function(self, pred, label):
    return F.cross_entropy(pred, label, ignore_index=0)


  def encoding(self, inputs, h0=None, lens=None):

    if lens:
      #inputs: NxLxC
      sorted_indices = np.argsort(lens.cpu().numpy())[::-1].tolist()
      lens = lens[sorted_indices]
      wemb = inputs[sorted_indices]
  
      #x: NxLxC
      x = wemb.transpose(0,1)
      #x: LxNxC
      bsize = x.size()[1]
      embs = torch.nn.utils.rnn.pack_padded_sequence(x, lens)
      output, hn = self.en(embs)
  
      seqs, lens = torch.nn.utils.rnn.pad_packed_sequence(output)
      lens = torch.from_numpy(np.asarray(lens).reshape(bsize,1)).float().cuda()
      seqs = seqs.transpose(0,1) #NxLxC
  
      unsorted_indices = np.argsort(sorted_indices).tolist()
      seqs = seqs[unsorted_indices]

    else:
      embs = inputs.transpose(0,1)
      output, hn = self.en(embs, h0)
      seqs = output.transpose(0,1)

    return seqs, hn


  def forward(self, tokens, h0=None):

    # Encoding
    wembs = self.embed(tokens[:, :self.config.seq_length]) 

    output, hn = self.encoding(wembs, h0)
    pred = self.proj(output)
    pred = self.out(pred)

    loss = self.loss_function(pred.view(-1, self.config.n_embed), tokens[:, 1:].contiguous().view(-1))

    return loss.view(1)

