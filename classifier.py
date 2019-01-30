import torch 
import torch.nn as nn


class classifier(nn.Module):

  def __init__(self, config):
    super(classifier, self).__init__()
    self.config = config

    d_z = config.d_hidden * (2 if config.birnn else 1) * 4

    inner_dim = config.d_hidden

    if config.classifier == "log_reg":
      self.clser = nn.Linear(d_z, 1)
    elif config.classifier == "mlp":
      self.clser = nn.Sequential(
                   nn.Linear(d_z, inner_dim),    
                   nn.ReLU(True),
                   nn.Linear(inner_dim, 1 if config.n_classes == 2 else config.n_classes)
                 )

    self.init_weights()


  def init_weights(self):

    if self.config.classifier == "mlp":
      for l in self.clser:
        if isinstance(l, nn.Linear):
          torch.nn.init.kaiming_uniform_(l.weight.data)
          l.bias.data.zero_()
    else:
      torch.nn.init.kaiming_uniform_(self.clser.weight.data)
      self.clser.bias.zero_()


  def forward(self, pre, hypo):

    z = torch.cat([pre, hypo, torch.abs(pre-hypo), pre*hypo], dim=1)
    pred = self.clser(z)

    return pred
    
