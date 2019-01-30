# TPRU
The implementation of our proposed TPRU with the code for experiments on Logical Entailment task
* [Learning Distributed Representations of Symbolic Structure Using Binding and Unbinding Operations](https://arxiv.org/pdf/1810.12456.pdf) - Shuai Tang, Paul Smolensky, Virginia R. de Sa

The model and the results in the paper are currently out-dated and we plan to update the paper soon.

## Getting started
The repo contains the code of our proposed TPRU, which is a recurrent unit based on Tensor Product Representations, and the code for experiments on Logical Entailment task. Both plain architecture and BiDAF architecture are provided here. 

### Data preparation
The code for downloading and preprocessing the data for Logical Entailment task is provided.
```
cd data/logical-entailment/
sh download.sh
```

### Training
```
CUDA_VISIBLE_DEVICES=0 python -u main.py --birnn --cuda 
 --lr 1e-3 --epochs 90 \
 --batch_size 64 \
 --d_hidden 64 \
 --n_layers 1 \
 --model_type plain \
 --n_roles 256 2>&1 | tee training.log 
```


### Contact
* [shuaitang93@ucsd.edu](mailto:shuaitang93.ucsd.edu) - Email
* [@Shuai93Tang](https://twitter.com/Shuai93Tang) - Twitter
* [Shuai Tang](http://shuaitang.github.io/) - Homepage
