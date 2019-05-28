import os
import time
import sys

import numpy as np
import json
import h5py
import copy
from tqdm import *

import argparse
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.optim import Adam

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

sys.path.insert(0, 'models/')
import lm

def get_batch():

  randind = np.random.randint(0, len(corpus["train"])-config.seq_length-1, config.batch_size)
  indices = np.array(list(map(L, randind)))
  tokens = corpus["train"][indices]

  tokens = T(tokens)
  # bsize x (seq_length + 1) 
  return tokens 


def loss_function( pred, labels ):

  return F.cross_entropy(pred, labels, ignore_index=0)


def save_model(model):  
  model2save = copy.deepcopy(model)
  filename = "wiktext-103k_100k_2layers_TPRU_dim{:04d}_roles{:04d}_seq{:04d}_".format(config.d_hidden, config.n_roles, config.seq_length)
  snapshot_prefix = os.path.join(config.save_path + "/", filename)
  torch.save(model2save, snapshot_prefix + "_best.pt")


def train(epoch):

  model.train()

  iters = np.ceil(len(corpus["train"])*1./config.batch_size/config.seq_length).astype(int)
  start = time.time() 
  total_loss = 0.

  dtime = 0.

  for i in range(iters):

    dbegin = time.time()
    tokens = get_batch()
    dtime += time.time() - dbegin

    out = model(tokens)
    
    loss = out.mean()
    total_loss += loss.item()   

    opt.zero_grad()
    loss.backward()  
    
    torch.nn.utils.clip_grad_value_(params, config.clip)
    opt.step()
  
    if i % 50 == 0:
      print("Train T {:6.2f}, Data {:6.2f}, E {:2d}, I {:6d}/{:6d}, L {:.2f}".format(time.time()-start, dtime, epoch, i, iters, loss.item()))
      start = time.time()
      dtime = 0.

  return total_loss / iters



def evaluate(cmodel, split, epoch):
  
  cmodel.eval()

  seq_length = int(config.batch_size * config.seq_length / 16)
  data = corpus[split]
  total_length = len(data)
  print("Number of Tokens in " + split + " : " + str(total_length))
  iters = np.ceil(total_length*1./seq_length).astype(int)
  start = time.time()
  total_loss = 0.
  dtime = 0.

  h0 = None

  with torch.no_grad():
    for i in range(iters): 
      dbegin = time.time()
      start_idx = i * seq_length
      end_idx = start_idx + seq_length
      end_idx = end_idx if end_idx < total_length-1 else total_length-1

      tokens = T(data[start_idx:end_idx+1])
      embed = cmodel.embed(tokens[:-1]).unsqueeze(0)
   
      masks = torch.ones_like(embed)[:,:,0:1]      

      seq, h0 = cmodel.encoding(embed, masks, h0)

      seq = cmodel.proj(seq)
      seq = cmodel.out(seq)

      loss = F.cross_entropy(seq.view(-1, config.n_embed), tokens[1:], ignore_index=0)
      total_loss += loss.item() * (end_idx - start_idx)
      if i % 50 :  
        print("Eval T {:6.2f}, data {:6.2f}, E {:2d}, I {:6d}/{:6d}, L {:.2f}".format(time.time()-start, dtime, epoch, i, iters, loss.item()))
        start = time.time()
        dtime = 0.
  
  return total_loss / total_length


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='PyTorch')
  parser.add_argument("--datafile", type=str,
                      default="wiki.bpe.h5",
                      help="hdf5 file")
  parser.add_argument("--dictfile", type=str,
                      default="wiki.train.bpe.tokens.dict.json",
                      help="vocabulary")
  parser.add_argument('--datapath', type=str, 
                      default="data/wikitext-103/bpe/",
                      help='location of the data corpus')
  parser.add_argument('--save_path', type=str, 
                      default='trained_models/',
                      help='path to save model')
  
  parser.add_argument('--seed', type=int, default=1111,
                      help='random seed')
  
  parser.add_argument('--d_embed', type=int, default=512)
  parser.add_argument('--d_hidden', type=int, default=512)
  parser.add_argument('--n_heads', type=int, default=8)
  parser.add_argument('--n_roles', type=int, default=16)
  parser.add_argument('--n_layers', type=int, default=2)
  parser.add_argument('--n_slices', type=int, default=8)
  
  parser.add_argument('--bidirectional', action='store_true')
  parser.add_argument('--cuda', action='store_true')
  parser.add_argument('--tie_weights', action='store_true')
  
  parser.add_argument('--seq_length', type=int, default=128)
  parser.add_argument('--max_lr', type=float, default=2.5e-4)
  parser.add_argument('--min_lr', type=float, default=0)
  parser.add_argument('--clip', type=float, default=10.)
  parser.add_argument('--batch_size', type=int, default=16)
  parser.add_argument('--epochs', type=int, default=64)
  
  config = parser.parse_args()
  
  # Set the random seed manually for reproducibility.
  torch.manual_seed(config.seed)
  np.random.seed(config.seed)
  if torch.cuda.is_available():
      if not config.cuda:
          print("WARNING: You have a CUDA device, so you should probably run with --cuda")
      else:
          torch.cuda.manual_seed(config.seed)
  
  
  # Load corpus and dictionary
  f = h5py.File(config.datapath + "/" + config.datafile, "r")
  corpus = {}
  for split in ["train", "valid", "test"]:
    corpus[split] = np.copy(f[split])
  f.close()
  
  with open(config.datapath + "/" + config.dictfile, "r", encoding="utf-8") as f:
    vocab = json.load(f)
  config.n_embed = len(vocab)
  
  # Helper functions
  
  T = lambda x: torch.from_numpy(x).long().cuda()
  L = lambda x: np.arange(x, x+config.seq_length+1)

  # Model
  model = lm.model(config)
  devices = torch.cuda.device_count()
  device_ids = [i for i in range(devices)]
  output_device = device_ids[0]
  
  model = nn.DataParallel(model, device_ids, output_device)
  model = model.cuda() if config.cuda else model
  
  config.batch_size *= devices  
  params = filter(lambda p: p.requires_grad, model.parameters())
  
  opt = Adam(params, lr=config.max_lr, amsgrad=True)
  
  print('Finished setting up model')

  # Training
  best_loss = 100.
  cnt = 0
#  bestmodel = copy.deepcopy(model.module)

  for epoch in range(config.epochs):
    total_loss = train(epoch)
    print("Epoch {:6d}, Train Loss {:6.2f}".format(epoch, total_loss))
    eval_loss = evaluate(model.module, "valid", epoch)
    print("Valid Loss: {:3f}".format(eval_loss))

    if eval_loss < best_loss:
#      save_model(model.module)
      test_loss = evaluate(model.module, "test", config.epochs)
      print("Test Loss: {:3f}".format(test_loss))
      
      cnt = 0
    else:
      for param_group in opt.param_groups:
        config.lr = config.lr / 2. if config.lr > config.min_lr else config.min_lr        
        param_group['lr'] = config.lr
      cnt += 1

    if cnt == 3:
      break        


