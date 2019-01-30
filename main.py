import os
import sys
import time
import glob

import numpy as np
import json
import copy
from tqdm import *
from sklearn.utils import shuffle
import argparse
import time
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False

sys.path.insert(0, 'models/')

import plain
import bidaf

## data augmentation based on alpha-equivalence
def augmentation(sent, randint):

  pos = np.logical_and(sent >= ind_a, sent <= ind_a + 26 - 1)
  var = (sent - ind_a) * pos
  per = randint[var]
  return (per + ind_a) * pos + sent * (1 - pos)


## sample a batch of data 
def get_batch(i, split='train'):
  
  p = i*config.batch_size
  q = p + config.batch_size
  if q > cnt[split]:
    q = cnt[split]

  t_pre, t_hypo, labels, l_pre, l_hypo = [], [], [], [], []
  for x in data[split][p:q]:
    [l.append(c) for l, c in zip([t_pre, t_hypo, labels, l_pre, l_hypo],x)]
  
  m_pre, m_hypo = max(l_pre), max(l_hypo)
  m = max(m_pre, m_hypo)
  pre  = np.zeros((q-p, m), dtype=int)
  hypo = np.zeros((q-p, m), dtype=int)

  for a, b, l in zip(pre, t_pre, l_pre):
    for j in range(l):
      a[j] = b[j]

  for a, b, l in zip(hypo, t_hypo, l_hypo):
    for j in range(l):
      a[j] = b[j]


  if split == "train":
    randint = np.random.randint(0,26,size=26)
    pre = augmentation(pre, randint)
    hypo = augmentation(hypo, randint)


  if not config.cuda:
    print("Not Working")
    exit()
  else:
    labels = T(np.array(labels)).float().cuda()
    pre, hypo = T(pre).long().cuda(), T(hypo).long().cuda()

  return pre, hypo, l_pre, l_hypo, labels


## classification loss
def loss_function( pred, labels ):
  return F.binary_cross_entropy_with_logits(pred, labels)
  

## save best model
def save_model(model):
  info = "TPRU_logic_" + config.pooling + "dim{:03d}_roles{:03d}_bs{:03d}_".format(config.d_hidden, config.n_roles, config.batch_size)
  snapshot_prefix = os.path.join(config.save_path, info)
  snapshot_prefix += "_".join([config.pooling, "{:.2}".format(config.dp_ratio)])
  torch.save(model, snapshot_prefix + "_best.pt")


## evaluation
def evaluate(cmodel, split):

  cmodel.eval()
  iters = np.ceil(cnt[split]*1./config.batch_size).astype(int)
  num_correct = 0.
  start = time.time()
  
  with torch.no_grad():
    for i in range(iters):
      t_pre, t_hypo, l_len, l_hypo, labels = get_batch(i, split)
      pred = cmodel(t_pre, t_hypo, l_len, l_hypo).view(-1)
      pred_labels = F.relu(torch.sign(pred))
      num_correct += torch.sum(pred_labels == labels).data.item()
  return pred_labels, num_correct * 100. / cnt[split]


## shuffle the training data every iteration
def data_shuffling():
  data["train"] = shuffle(data["train"], random_state=config.seed) 

 
## training
def train(epoch):

  model.train()
  model.embed.eval()
  data_shuffling()

  iters = np.ceil(cnt['train']*1./config.batch_size).astype(int)
  num_correct = 0.
  start = time.time() 
  d_time = 0.

  for i in range(iters):

    begin = time.time()
    t_pre, t_hypo, l_pre, l_hypo, labels = get_batch(i, 'train')
    d_time += time.time() - begin    

    pred = model(t_pre, t_hypo, l_pre, l_hypo).view(-1)
    loss = loss_function(pred, labels)
   
    opt.zero_grad()
    loss.backward()  
    
    torch.nn.utils.clip_grad_value_(params, config.clip)
    opt.step()
  
    with torch.no_grad():
      pred_labels = F.relu(torch.sign(pred))
      correct = torch.sum(pred_labels == labels).item()
      num_correct += correct
      acc = correct * 100. / len(labels)

    if i % 100 == 0:
      print("T {:6.2f}, D {:6.2f}, E {:2d}, I {:6d}/{:6d}, L {:.2f}, acc {:2.2f}".format(time.time()-start, d_time, epoch, i, iters, loss.item(), acc))
      start = time.time()
      d_time = 0.

  return num_correct * 100. / cnt['train'] 



if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='PyTorch')
  parser.add_argument('--seed', type=int, default=1111,
                      help='random seed')
  parser.add_argument('--save_path', type=str, 
                      default='trained_models/',
                      help='path to save model')
  parser.add_argument('--data_path', type=str,
                      default='data/logical-entailment/preprocessed/')

  parser.add_argument('--d_embed', type=int, default=64)
  parser.add_argument('--d_hidden', type=int, default=64)  
  parser.add_argument('--n_heads', type=int, default=8)
  parser.add_argument('--n_roles', type=int, default=4)
  parser.add_argument('--n_layers', type=int, default=1)

  parser.add_argument('--classifier', type=str, default="mlp")
  parser.add_argument('--pooling', type=str, default="max")
  parser.add_argument('--model_type', type=str, default="plain") 


  parser.add_argument('--dp_ratio_rnn', type=float, default=0.00)
  parser.add_argument('--dp_ratio', type=float, default=0.00)
  parser.add_argument('--dp_ratio_semb', type=float, default=0.00)
  parser.add_argument('--birnn', action='store_true')
  parser.add_argument('--cuda', action='store_true')
  parser.add_argument('--tie_weights', action='store_true')
  parser.add_argument('--seq_length', type=int, default=40)
  parser.add_argument('--lr', type=float, default=5e-4)
  parser.add_argument('--min_lr', type=float, default=1e-6)
  parser.add_argument('--clip', type=float, default=10.)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--save_every', type=int, default=20000)
  parser.add_argument('--epochs', type=int, default=10)
  parser.add_argument('--beta', type=float, default=0.0)
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
  
  splits = ['test_hard', 'train', 'test_easy', 'validate', 'test_exam', 'test_massive', 'test_big'] 
  test_splits = ['test_hard', 'test_easy', 'test_exam', 'test_massive', 'test_big']

  with open(config.data_path + "data.json", "r") as f:
    data = json.load(f)
  
  cnt = {}
  for split in splits:
    cnt[split] = len(data[split])
  
  with open(config.data_path + "dict.json", "r") as f:
    vocab = json.load(f)
  token2idx = vocab["token2idx"]
  idx2token = vocab["idx2token"]
  config.n_embed = len(token2idx)
  
  print('Finished loading training corpus')
 
  # helper function   
  T = lambda x: torch.from_numpy(x)
  ind_a = token2idx["a"]

  # Define model
  model = bidaf.model(config) if config.model_type == "bidaf" else plain.model(config)
  model = model.cuda() if config.cuda else model
  params = filter(lambda p: p.requires_grad, model.parameters())
  
  opt = Adam(params, lr=config.lr, amsgrad=True)
  
  print('Finished setting up model')

  best_train_acc = 0.
  best_dev_acc = 0.
  bestmodel = copy.deepcopy(model)

  begin = time.time()

  for epoch in range(config.epochs):
    train_acc = train(epoch)
    _, dev_acc = evaluate(model, "validate")
    print("Train Acc {:2.2f}, Dev Acc {:2.2f}".format(train_acc, dev_acc))

    if dev_acc > best_dev_acc:    
      best_dev_acc = dev_acc
      best_train_acc = train_acc
      for split in test_splits:
        print(split + " : {:2.2f}".format(evaluate(model, split)[1]))
      bestmodel = copy.deepcopy(model)      
      save_model(bestmodel)
 
  print("Best train acc: {:2.2f}, Best dev acc: {:2.2f}".format(best_train_acc, best_dev_acc))
  for split in test_splits:
    print(split + " : {:2.2f}".format(evaluate(bestmodel, split)[1]))

