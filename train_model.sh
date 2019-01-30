python -u main.py --f_enable --b_enable --cuda --lr 1e-3 --epochs 90  \
 --batch_size 64 \
 --d_hidden 64 \
 --n_roles 512   2>&1 | tee training.log 
