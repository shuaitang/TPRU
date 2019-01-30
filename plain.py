import torch 
import torch.nn as nn
import numpy as np

from torch.nn import functional as F

from RNN import TPRU, TPRRNN
from classifier import classifier


class model(nn.Module):

  def __init__(self, config):
    super(model, self).__init__()
    self.config = config
    self.hidden_size = config.d_hidden 

    self.embed = nn.Embedding(config.n_embed, config.d_embed, padding_idx=0)
    self.en = TPRRNN(self.config)
#    self.dropout = nn.Dropout(self.config.dp_ratio_semb) if config.dp_ratio_semb else lambda x: x
    self.clser = classifier(self.config)    



  def init_weights(self):
    torch.nn.init.orthogonal_(self.embed.weight.data)


  def rep_pooling(self, seqs, lasth, lens):
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


  def encoding(self, inputs, lens):

    #inputs: NxLxC
    sorted_indices = np.argsort(lens)[::-1].tolist()
    lens = np.array(lens)[sorted_indices].tolist()
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
    z = self.rep_pooling(seqs, hn, lens)

    unsorted_indices = np.argsort(sorted_indices).tolist()
    z = z[unsorted_indices]

    return z


  def forward(self, t_pre, t_hypo, l_pre, l_hypo):

    # Encoding
    wemb = self.embed(torch.cat([t_pre, t_hypo], dim=0))
    lens = l_pre + l_hypo

    u, v = self.encoding(wemb, lens).chunk(2,0)

    # Classification
    pred = self.clser(u, v)

    return pred   
