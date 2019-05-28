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

def label_mapping(label):
  if label == 'entailment':
    return 1
  elif label == 'neutral' or label == 'hidden':
    return 0
  elif label == 'contradiction':
    return 2
  else:
    return 0


def get_batch(i, split='train'):
  
  p = i*config.batch_size
  q = p + config.batch_size
  if q > num[split]:
    q = num[split]

  t_pre = samples[split]["s1"][p:q]
  t_hypo = samples[split]["s2"][p:q]
  labels = samples[split]["label"][p:q]

  if not config.cuda:
    print("Not Working")
    exit()
  else:
    labels = T(np.array(labels))
 
  return t_pre, t_hypo, labels


def loss_function( pred, labels ):

  return  F.cross_entropy(pred, labels)


def save_model(model):
  snapshot_prefix = os.path.join(config.save_path, 'TPRU_MNLI_'+config.pooling+"{:04d}_{:04d}_bs{:03d}".format(config.d_hidden, config.n_roles, config.batch_size)+"_")
  torch.save(model.en, snapshot_prefix + "_en_best.pt")
  torch.save(model.clser, snapshot_prefix + "_clser_best.pt")


def evaluate(split):

  model.eval()
  iters = np.ceil(num[split]*1./config.batch_size).astype(int)
  num_correct = 0.
  start = time.time()
  
  with torch.no_grad():
    for i in range(iters):
      t_pre, t_hypo, labels = get_batch(i, split)
      pred = model(t_pre, t_hypo)
      pred_labels = torch.max(pred, dim=1)[1] 
      num_correct += torch.sum(pred_labels == labels).data.item()
      if i % 50 == 0:
        print("T {:6.2f}, I {:6d}/{:6d}".format(time.time()-start, i, iters))
        start = time.time()
  return pred_labels, num_correct * 100. / num[split]


def data_shuffling(): 
  samples["train"]["s1"], samples["train"]["s2"], samples["train"]["label"], samples["train"]["genre"] =  shuffle(samples["train"]["s1"], samples["train"]["s2"], samples["train"]["label"], samples["train"]["genre"], random_state=config.seed)

  

def train(epoch):

  model.train()
  data_shuffling()

  iters = np.ceil(num['train']*1./config.batch_size).astype(int)
  num_correct = 0.
  start = time.time() 

  for i in range(iters):

    t_pre, t_hypo, labels = get_batch(i, 'train')
    pred = model(t_pre, t_hypo)
    loss = loss_function(pred, labels)
    opt.zero_grad()
    loss.backward()  
    
    torch.nn.utils.clip_grad_value_(params, config.clip)
    opt.step()
  
    with torch.no_grad():
      pred_labels = torch.max(pred, dim=1)[1]
      correct = torch.sum(pred_labels == labels).item()
      num_correct += correct
      acc = correct * 100. / len(labels)

    if i % 50 == 0:
      print("T {:6.2f}, E {:2d}, I {:6d}/{:6d}, L {:.2f}, acc {:2.2f}".format(time.time()-start, epoch, i, iters, loss.item(), acc))
      start = time.time()

  return num_correct * 100. / num['train'] 


if __name__ == "__main__":


  parser = argparse.ArgumentParser(description='PyTorch')

  parser.add_argument('--seed', type=int, default=1111,
                      help='random seed')
  parser.add_argument('--data_path', type=str, 
                      default='data/mnli/preprocessed/')
  parser.add_argument('--save_path', type=str, 
                      default='trained_models/',
                      help='path to save model')
  parser.add_argument('--elmo_path', type=str,
                      default='data/elmo/')
  parser.add_argument('--elmo', action='store_true')
  parser.add_argument('--finetune_elmo', action='store_true')

  parser.add_argument('--d_hidden', type=int, default=512)
  
  parser.add_argument('--n_slices', type=int, default=1)
  parser.add_argument('--n_roles', type=int, default=4)
  parser.add_argument('--n_classes', type=int, default=3)

  parser.add_argument('--model_type', type=str, default="plain")
  parser.add_argument('--classifier', type=str, default="mlp")
  parser.add_argument('--pooling', type=str, default="max")
  

  parser.add_argument('--n_layers', type=int, default=1)
  parser.add_argument('--bidirectional', action='store_true')

  parser.add_argument('--cuda', action='store_true')

  parser.add_argument('--seq_length', type=int, default=40)
  parser.add_argument('--lr', type=float, default=5e-4)
  parser.add_argument('--min_lr', type=float, default=1e-6)
  parser.add_argument('--clip', type=float, default=10.)
  parser.add_argument('--batch_size', type=int, default=64)
  parser.add_argument('--save_every', type=int, default=20000)
  parser.add_argument('--epochs', type=int, default=10)

  config = parser.parse_args()
  
  config.d_embed = (2048 if config.model_type == "plain" else 1024) if config.elmo else config.d_embed
  
  # Set the random seed manually for reproducibility.
  torch.manual_seed(config.seed)
  np.random.seed(config.seed)
  if torch.cuda.is_available():
    if not config.cuda:
      print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
      torch.cuda.manual_seed(config.seed)

  # Load corpus and dictionary

  splits = ["train", "dev_matched", "dev_mismatched", "test_matched", "test_mismatched", "test"]

  samples = {}
  num = {}
  for split in splits:
    samples[split] = {"s1":[], "s2":[], "label":[], "genre":[]}
    with open(config.data_path + split + ".jsonl") as f:
      data = f.readlines()
    data = list(map(json.loads, data))
    num[split] = len(data)
    [samples[split][key].append(d[key]) for d in data for key in samples[split].keys()]
    samples[split]["label"] = list(map(label_mapping, samples[split]["label"]))
    samples[split]["s1"] = [x[:config.seq_length] for x in samples[split]["s1"]]
    samples[split]["s2"] = [x[:config.seq_length] for x in samples[split]["s2"]]
  
  print('Finished loading training corpus')
  
  # Training Batch
  
  T = lambda x: torch.from_numpy(x).long().cuda()

  # Model
  model = bidaf.model(config) if config.model_type == "bidaf" else plain.model(config) 
  model = model.cuda() if config.cuda else model 

  params = filter(lambda p: p.requires_grad, model.parameters())
  
  opt = Adam(params, lr=config.lr, amsgrad=True)
  
  print('Finished setting up model')
  
  # Training code

  best_train_acc = 0.
  best_dev_matched_acc = 0.
  best_dev_mismatched_acc = 0.
  best_avg_acc = 0.
  dev_avg_acc = 0.
  bestmodel = copy.deepcopy(model)

  for epoch in range(config.epochs):
    train_acc = train(epoch)
    _, dev_matched_acc = evaluate("dev_matched")
    _, dev_mismatched_acc = evaluate("dev_mismatched")
    dev_avg_acc = 0.5 * (dev_matched_acc + dev_mismatched_acc)
    print("Train Acc {:2.2f}, Dev Matched Acc {:2.2f}, Dev Mismatched Acc {:2.2f}".format(train_acc, dev_matched_acc, dev_mismatched_acc))

    if dev_avg_acc > best_avg_acc:    
      best_avg_acc = dev_avg_acc
      best_train_acc = train_acc
      best_dev_matched_acc = dev_matched_acc
      best_dev_mismatched_acc = dev_mismatched_acc
      bestmodel = copy.deepcopy(model)
      save_model(bestmodel)
 
    print("Best train acc: {:2.2f}, Best dev matched acc: {:2.2f}, Best dev mismatched acc: {:2.2f}".format(best_train_acc, best_dev_matched_acc, best_dev_mismatched_acc))
  

