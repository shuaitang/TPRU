CUDA_VISIBLE_DEVICES=0 python -u main.py --birnn --cuda --lr 1e-3 --epochs 50 \
 --batch_size 64 \
 --d_hidden 64 \
 --n_layers 1 \
 --n_roles 256   2>&1 | tee training.log 
