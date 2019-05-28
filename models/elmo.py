from allennlp.modules.elmo import Elmo, batch_to_ids
import torch.nn as nn
import os

class ELMoEmbedding(nn.Module):

  def __init__(self, config):
    super(ELMoEmbedding, self).__init__()
    options_file = config.elmo_path + "/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = config.elmo_path + "/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
 
    assert os.path.isfile(options_file) and os.path.isfile(weight_file), "Set PATHs"

    self.elmo = Elmo(options_file, weight_file, 2, dropout=0, requires_grad=config.finetune_elmo)

    if config.cuda:
      self.T = lambda x: x.long().cuda()
    else:
      self.T = lambda x: x.long()

  def forward(self, sentences):
    char_ids = self.T(batch_to_ids(sentences))
    embeddings = self.elmo(char_ids)
#    lens = embeddings['mask'].sum(1)

    return embeddings['elmo_representations'], embeddings['mask']
