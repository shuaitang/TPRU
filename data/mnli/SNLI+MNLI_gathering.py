import json
import numpy as np
from pandas import read_table
import sys
import re


prefix = "nli/"
trainfiles = [  "SNLI/original/snli_1.0_train.jsonl",
		"SNLI/original/snli_1.0_dev.jsonl",
		"MNLI/original/multinli_1.0_train.jsonl"]
devfiles = ["MNLI/original/multinli_1.0_dev_matched.jsonl",
            "MNLI/original/multinli_1.0_dev_mismatched.jsonl"]
testfiles = ["MNLI/test_matched.tsv", 
             "MNLI/test_mismatched.tsv",
             "SNLI/test.tsv"]

savepath = "preprocessed/"


def tokenize(string):
  string = re.sub(r'\(|\)', '', str(string))
  return string.split()


def load_jsonl(line):
  try:
    data = json.loads(line)
  except:
    data = json.loads(line.decode("utf-8"))
  return {"label": data["gold_label"],
          "s1": tokenize(data["sentence1_binary_parse"]),
          "s2": tokenize(data["sentence2_binary_parse"]),
          "genre": data["genre"] if "genre" in data.keys() else "snli"}
  

def main():
  # training:
  training_data = []
  for name in trainfiles:
    with open(prefix + name, "r") as f:
      lines = f.readlines()
    training_data.extend(list(map(load_jsonl, lines)))
  print("Number of training pairs: {:10d}".format(len(training_data)))
  with open(savepath + "train.jsonl", "w") as f:
    for d in training_data:
      f.write(json.dumps(d)+"\n")

  # dev:
  dev_data = []
  for name in devfiles:
    with open(prefix + name, "r") as f:
      lines = f.readlines()
    split = name.split("/")[2].split("_1.0_")[1]
    dev_data = list(map(load_jsonl, lines))
    print("Number of " + split + " pairs: " + str(len(dev_data)))
    with open(savepath + split, "w") as f:
      for d in dev_data:
        f.write(json.dumps(d)+"\n")

  # test:
  test_data = []
  for name in testfiles:
    table = read_table(prefix + name, sep="\t")
    s1 = list(table["sentence1_binary_parse"])[1:]
    s1 = list(map(tokenize, s1))
    s2 = list(table["sentence2_binary_parse"])[1:]
    s2 = list(map(tokenize, s2))
    try:
      genre = list(table["genre"])[1:]
    except:
      genre = ["snli"] * len(s1)
    split = name.split("/")[1].split(".tsv")[0]
    print("Number of " + split + " pairs: " + str(len(genre)))
    with open(savepath + split + ".jsonl", "w") as f:
      for pre, hypo, g in zip(s1, s2, genre):
        f.write(json.dumps({"label": 'hidden', "s1":pre, "s2":hypo, "genre":g}) + "\n")
    
if __name__=="__main__":
  sys.exit(main())
