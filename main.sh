rddm3: lr 8e-5   bs:2  sample_step:2
rddm_mamba: lr 8e-5   bs:1  sample_step:2


rddm_mamba_clipiqa:lr 8e-5   bs:1  sample_step:2  total 4000000


rddm_dn_all:所有数据 lr 8e-5   bs:1  sample_step:2  total 2000000


assm1   4000000 assm +
asss2   4000000 assm *
assm3   2000000 assm+cross-attention
assm4   2000000  assm

CUDA_VISIBLE_DEVICES=2 python train.py --name rddm_all_ASSM_doseclip_rnc_sup_mayo_newhead3 --is_train

#rddm single
CUDA_VISIBLE_DEVICES=6 python train.py --name rddm_d2 --is_train --train_num_steps 100000