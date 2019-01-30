import numpy
import json
import glob

filepath = "logical-entailment-dataset/data/"
filenames = glob.glob(filepath + "*.txt")
filenames = [x.split("/data/")[1].split(".txt")[0] for x in filenames]

savepath = "preprocessed/"


data = {split:[] for split in filenames}
props = []
for split in filenames:
  path = filepath + split + ".txt"
  with open(path, "r") as f:
    d = f.readlines()
  data[split] = [x.strip().split(",")[:3] for x in d]
  pre = [x[0] for x in data[split]]
  hypo = [x[1] for x in data[split]]
  props.extend(pre)
  props.extend(hypo)
  print(split + " " + str(len(d)))

tokens = []
lens = []
[tokens.extend(list(x)) for x in props]
[lens.append(len(x)) for x in props]
print("maximum length is: " + str(max(lens)))

unique_tokens = sorted(list(set(tokens)))
print(unique_tokens)

token2idx = {x:i+1 for i, x in enumerate(unique_tokens)}
token2idx["<pad>"] = 0
idx2token = {idx: token for token, idx in token2idx.items()}

with open(savepath + "dict.json", "w") as f:
  json.dump({"token2idx":token2idx, "idx2token": idx2token}, f)

data2save = {}
E = lambda l: [token2idx[x] for x in list(l)]

for split in filenames:
  data2save[split] = []
  for pair in data[split]:
    data2save[split].append([E(pair[0]), E(pair[1]), int(pair[2]), len(E(pair[0])), len(E(pair[1]))])

with open(savepath + "data.json", "w") as f:
  json.dump(data2save, f)  

