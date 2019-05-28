# TPRU
The implementation of our proposed TPRU with the code for experiments on Logical Entailment task, Multi-genre Natural Language Inference and Language Modelling. Both plain and BiDAF architectures are provided here.
* [Learning Distributed Representations of Symbolic Structure Using Binding and Unbinding Operations](https://arxiv.org/pdf/1810.12456v5.pdf) - Shuai Tang, Paul Smolensky, Virginia R. de Sa

## Requirements
```
pytorch >= 1.0
python >= 3.5
```

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
CUDA_VISIBLE_DEVICES=0 python -u main_logical.py --bidirectional --cuda \
 --lr 1e-3 --epochs 90 \
 --batch_size 64 \
 --d_hidden 64 \
 --n_layers 1 \
 --model_type plain \
 --n_roles 256 2>&1 | tee training_logical.log 
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
CUDA_VISIBLE_DEVICES=0 python -u main_mnli.py --bidirectional --cuda --elmo \
 --lr 1e-4 --epochs 15 \
 --batch_size 64 \
 --d_hidden 512 \
 --n_layers 1 \
 --model_type plain \
 --n_roles 1024 2>&1 | tee training_mnli.log
```


## Language Modelling on WikiText-103

The code here conducts data preprocessing and training a language modelling with our proposed TPRU on WikiText-103. It supports multi-GPU training.


### Data preparation
The wikitext-103 is encoded by Byte Pair Encoding method. The encoded corpus is saved as a HDF5 file.

```
cd data/wikitext-103
sh 1_download.sh
sh 2_processing.sh 100000
python 3_conversion.py
```


### Training

```
CUDA_VISIBLE_DEVICES=0 python -u main_lm.py --cuda \
--batch_size 32 \
--seq_length 128 \
--d_hidden 1024 \
--n_roles 1024 \
--n_layers 2 2>&1 | tee training_lm.log
```


## Contact
* [shuaitang93@ucsd.edu](mailto:shuaitang93.ucsd.edu) - Email
* [@Shuai93Tang](https://twitter.com/Shuai93Tang) - Twitter
* [Shuai Tang](http://shuaitang.github.io/) - Homepage
