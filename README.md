# TPRU
The implementation of our proposed TPRU with the code for experiments on Logical Entailment task
* [Learning Distributed Representations of Symbolic Structure Using Binding and Unbinding Operations](https://arxiv.org/pdf/1810.12456.pdf) - Shuai Tang, Paul Smolensky, Virginia R. de Sa
The model and the results in the paper are currently out-dated and we plan to update the paper soon.

# Introduction

The repo contains the code of our proposed TPRU, which is a recurrent unit based on Tensor Product Representations, and the code for experiments on Logical Entailment task. Both plain architecture and BiDAF architecture are provided here. 

## Logical Entailment
Here come the instructions on training our TPRU on Logical Entailment task.


### Data preparation
The code for downloading and preprocessing the data for Logical Entailment task is provided.
```
cd data/logical-entailment/
sh download.sh
```

### Training
```
CUDA_VISIBLE_DEVICES=0 python -u main_logical.py --birnn --cuda \
 --lr 1e-3 --epochs 90 \
 --batch_size 64 \
 --d_hidden 64 \
 --n_layers 1 \
 --model_type plain \
 --n_roles 256 2>&1 | tee training.log 
```

## Multi-genre Natural Language Inference
The code for training our TPRU on MNLI is provided here.

### Data preparation
We recommend to learn models with ELMo embeddings, and here is a piece of code to download ELMo files.
```
cd data/elmo/
sh download.sh
```
Data preprocessing is carried out as follows:
```
cd data/mnli/
sh preprocessing.sh 
```


### Training
```
CUDA_VISIBLE_DEVICES=0 python -u main_mnli.py --birnn --cuda --elmo \
 --lr 1e-4 --epochs 15 \
 --batch_size 64 \
 --d_hidden 512 \
 --n_layers 1 \
 --model_type plain \
 --n_roles 1024 2>&1 | tee training.log
```


## Contact
* [shuaitang93@ucsd.edu](mailto:shuaitang93.ucsd.edu) - Email
* [@Shuai93Tang](https://twitter.com/Shuai93Tang) - Twitter
* [Shuai Tang](http://shuaitang.github.io/) - Homepage
