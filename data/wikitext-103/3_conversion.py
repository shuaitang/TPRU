import json
import h5py

datafile = "bpe/wiki"
dictfile = "bpe/wiki.train.bpe.tokens.dict.json"
savefile = "bpe/wiki.bpe.h5"

with open(dictfile, "r", encoding="utf-8") as f:
  vocab = json.load(f)

fw = h5py.File(savefile, "w")

for split in ["train", "valid", "test"]:
  filename = datafile + "." + split + ".bpe.tokens"

  with open(filename, "r", encoding="utf-8") as f:
    data = f.readlines()
  data = list(filter(lambda p: p != "", [x.strip() for x in data]))
  
  print("Finished loading data...")
  
  def encoding(sent):
    tokens = sent.split(" ")
    inds = []
    for token in tokens:
      try:
        ind = vocab[token]
      except:
        ind = vocab["<unk>"]
      inds.append(ind)
    inds.extend([vocab["<eos>"],vocab["<pad>"]])
  
    return inds
  
  sents = list(map(encoding, data))
  print("Finished mapping data...")
  
  indices = []
  [indices.extend(s) for s in sents]
  
  print("Finished concatenating data...")

  fw.create_dataset("/" + split, data=indices)
    
fw.close()



