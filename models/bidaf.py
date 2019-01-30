import torch 
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from RNN import TPRRNN
from classifier import classifier
from elmo import ELMoEmbedding


class model(nn.Module):

  def __init__(self, config):
    super(model, self).__init__()
    self.config = config
    self.hidden_size = config.d_hidden 

    self.embed = ELMoEmbedding(config) if config.elmo else nn.Embedding(config.n_embed, config.d_embed, padding_idx=0)
    self.en = TPRRNN(self.config)
    self.clser = classifier(self.config) 

    att_config = config
    att_config.d_embed = config.d_embed + config.d_hidden * 4
    self.att = TPRRNN(self.config)


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


  def encoding(self, en, inputs, lens):

    #inputs: NxLxC
    sorted_indices = np.argsort(lens)[::-1].tolist()
    lens = np.array(lens)[sorted_indices]
    wemb = inputs[sorted_indices]

    #x: NxLxC
    x = wemb.transpose(0,1)
    #x: LxNxC
    bsize = x.size()[1]
    embs = torch.nn.utils.rnn.pack_padded_sequence(x, lens)
    output, hn = en(embs)

    seqs, lens = torch.nn.utils.rnn.pad_packed_sequence(output)
    lens = torch.from_numpy(np.asarray(lens).reshape(bsize,1)).float().cuda()
    seqs = seqs.transpose(0,1) #NxLxC

    unsorted_indices = np.argsort(sorted_indices).tolist()
    seqs = seqs[unsorted_indices]

    return seqs


  def attn(self, u, v):
    # u, v: NxLxC
    H = u.bmm(v.transpose(1,2)) # NxLxL
    Au = F.softmax(H, dim=1)
    Av = F.softmax(H, dim=2)
    v_ = Av.bmm(v)
    u_ = Au.transpose(1,2).bmm(u)
    return u_, v_ 


  def forward(self, t_pre, t_hypo, l_pre=None, l_hypo=None):

    # Encoding
    if self.config.elmo:
      tokens = t_pre + t_hypo
      wemb, lens = self.embed(tokens)
      lens = lens.cpu().numpy()
    else:
      wemb = self.embed(torch.cat([t_pre, t_hypo], dim=0))
      lens = l_pre + l_hypo

    u, v = self.encoding(self.en, wemb[0] if self.config.elmo else wemb, lens).chunk(2,0)
    u_, v_ = self.attn(u, v)

    U = torch.cat([u, v_], dim=2)
    V = torch.cat([v, u_], dim=2)
    attn_input = torch.cat([U,V], dim=0)

    if self.config.elmo:
      attn_input = torch.cat([attn_input, wemb[1]], dim=2)
    else:
      attn_input = torch.cat([attn_input, wemb], dim=2)

    z = self.encoding(self.att, attn_input, lens)
    
    z = self.rep_pooling(z)
    z_pre, z_hypo = z.chunk(2,0)


    # Classification
    pred = self.clser(z_pre, z_hypo)

    return pred
    
