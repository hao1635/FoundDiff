import os, glob, shutil
import sys
from src.util import get_logger

from src.denoising_diffusion_pytorch import GaussianDiffusion
from src.residual_denoising_diffusion_pytorch_final3 import (ResidualDiffusion,
                                                      Trainer, Unet, UnetRes,
                                                      set_seed)
# from src.residual_denoising_diffusion_pytorch_abla import (ResidualDiffusion,
#                                                       Trainer, Unet, UnetRes,
#                                                       set_seed)

import ipdb
import argparse
def make_dir(path, refresh=False):
    
    """ function for making directory (to save results). """
    
    try: os.mkdir(path)
    except: 
        if(refresh): 
            shutil.rmtree(path)
            os.mkdir(path)

# init
#os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [5])

# export CUDA_VISIBLE_DEVICES=4
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
sys.stdout.flush()
set_seed(10)
debug = False




parser = argparse.ArgumentParser(description="一个简单的命令行参数示例")

# 添加参数
parser.add_argument('--name', type=str, required=True, help='输入文件路径')
parser.add_argument('--is_train', action='store_true', help='is_train')
parser.add_argument('--verbose', action='store_true', help='是否打印详细信息')
parser.add_argument('--sampling_timesteps', type=int, default=2, help='采样的时间步数')
parser.add_argument('--epoch', type=int, default=100, help='采样的时间步数')
parser.add_argument('--dataset', type=str, default='2020_seen', help='输入文件路径')
parser.add_argument('--train_num_steps', type=int, default=200000, help='采样的时间步数')
parser.add_argument('--train_batch_size', type=int, default=2, help='采样的时间步数')
# 解析命令行参数
opt = parser.parse_args()


if debug:
    save_and_sample_every = 2
    sampling_timesteps = 10
    sampling_timesteps_original_ddim_ddpm = 10
    train_num_steps = 200
else:
    save_and_sample_every = 1000
    sampling_timesteps = opt.sampling_timesteps
    sampling_timesteps_original_ddim_ddpm = 250
    train_num_steps = opt.train_num_steps

original_ddim_ddpm = False
if original_ddim_ddpm:
    condition = False
    input_condition = False
    input_condition_mask = False
else:
    condition = True
    input_condition = False
    input_condition_mask = False

if condition:
    # Image restoration  
    if input_condition:
        folder = ["xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_mask_test.flist"]
    else:
        # folder = ["xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
        #           "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
        #           "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
        #           "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist"]
        folder = ["xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_train.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_free_test.flist",
                  "xxx/dataset/ISTD_Dataset_arg/data_val/ISTD_shadow_test.flist"]

    train_batch_size = opt.train_batch_size
    num_samples = 1
    sum_scale = 0.01
    image_size = 256
else:
    # Image Generation 
    folder = 'xxx/CelebA/img_align_celeba'
    train_batch_size = 128
    num_samples = 64
    sum_scale = 1
    image_size = 64

# num_unet = 2
# objective = 'pred_res_noise'
# test_res_or_noise = "res_noise"


num_unet = 1
objective = 'pred_res'
test_res_or_noise = "res"

if original_ddim_ddpm:
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    )
    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        sampling_timesteps=sampling_timesteps_original_ddim_ddpm,
        loss_type='l1',            # L1 or L2
    )
else:
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=num_unet,
        condition=condition,
        input_condition=input_condition,
        objective=objective,
        test_res_or_noise = test_res_or_noise
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=image_size,
        timesteps=1000,           # number of steps
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        loss_type='l2',            # L1 or L2
        condition=condition,
        sum_scale=sum_scale,
        input_condition=input_condition,
        input_condition_mask=input_condition_mask,
        test_res_or_noise = test_res_or_noise
    )

if opt.is_train:
    checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/'+opt.name
    make_dir(checkpoint_folder)
else:
    #checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_mamba_doseclip_rnc_sup_mayo' ## epoch 390
    
    #checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_mamba_doseclip_rnc_sup_mayo' ## epoch 390
    # checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_emamba_doseclip_rnc_sup_mayo_newhead2' ##epoch 100
    # checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_newhead' ##epoch 100

    # checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_mayo' ## epoch 140
    # make_dir(checkpoint_folder)

    checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/'+opt.name

#make_dir(results_folder+'/sample')


trainer = Trainer(
    opt,
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=2e-4,#8e-5,
    train_num_steps=train_num_steps,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=False,                        # turn on mixed precision
    convert_image_to="RGB",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=True,
    num_unet=num_unet,
    checkpoint_folder=checkpoint_folder,
    is_train=opt.is_train,  
    train_logger=None,
)

# ipdb.set_trace()
# train
if opt.is_train:
    # trainer.train_logger = get_logger(checkpoint_folder+'/train_final.log')
    trainer.train()

# test
else:
    if not trainer.accelerator.is_local_main_process:
        pass
    else:
        trainer.load(opt.epoch)
                # trainer.set_results_folder(
                #     './results/test_timestep_'+str(sampling_timesteps))
        if opt.dataset=='2020_unseen':
            make_dir(checkpoint_folder+'/test_final_unseen_npy',refresh=True)
            trainer.train_logger = get_logger(checkpoint_folder+'/test_final_unseen.log')
            trainer.results_folder=checkpoint_folder+'/test_final_unseen_npy'
            trainer.test(last=True)
        if opt.dataset=='2020_seen':
            make_dir(checkpoint_folder+'/test_final_npy',refresh=True)
            trainer.train_logger = get_logger(checkpoint_folder+'/test_final.log')
            trainer.results_folder=checkpoint_folder+'/test_final_npy'
            trainer.test(last=True)
        if opt.dataset=='2016_unseen':
            make_dir(checkpoint_folder+'/test_final_2016_npy',refresh=True)
            trainer.train_logger = get_logger(checkpoint_folder+'/test_final_2016.log')
            trainer.results_folder=checkpoint_folder+'/test_final_2016_npy'
            trainer.test(last=True)


# is_train=True
# if is_train:
#     checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_emamba_doseclip_rnc_sup_mayo_newhead_g'
#     make_dir(checkpoint_folder)
# else:
#     #checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_mamba_doseclip_rnc_sup_mayo' ## epoch 390
    
#     #checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_mamba_doseclip_rnc_sup_mayo' ## epoch 390
#     checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_mamba_doseclip_rnc_sup_mayo_newhead' ##epoch 100
#     # checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_newhead' ##epoch 100

#     # checkpoint_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_mayo' ## epoch 140
#     # make_dir(checkpoint_folder)

# #make_dir(results_folder+'/sample')


# trainer = Trainer(
#     diffusion,
#     folder,
#     train_batch_size=train_batch_size,
#     num_samples=num_samples,
#     train_lr=2e-4,#8e-5,
#     train_num_steps=train_num_steps,         # total training steps
#     gradient_accumulate_every=2,    # gradient accumulation steps
#     ema_decay=0.995,                # exponential moving average decay
#     amp=False,                        # turn on mixed precision
#     convert_image_to="RGB",
#     condition=condition,
#     save_and_sample_every=save_and_sample_every,
#     equalizeHist=False,
#     crop_patch=False,
#     generation=True,
#     num_unet=num_unet,
#     checkpoint_folder=checkpoint_folder,
#     is_train=is_train,  
#     train_logger=None,
# )


# # train
# if is_train:
#     trainer.train_logger = get_logger(checkpoint_folder+'/train_final.log')
#     trainer.results_folder='/mnt/miah203/zhchen/DN_foundation/rddm_checkpoints/rddm_all_emamba_doseclip_rnc_sup_mayo_newhead2/sample'
#     trainer.load(300)
#     trainer.results_folder=checkpoint_folder+'/sample'
#     trainer.step=0
#     trainer.train()

# # test
# else:
#     if not trainer.accelerator.is_local_main_process:
#         pass
#     else:
#         trainer.load(200)
#                 # trainer.set_results_folder(
#                 #     './results/test_timestep_'+str(sampling_timesteps))
#         make_dir(checkpoint_folder+'/test_final_2016_npy',refresh=True)
#         trainer.train_logger = get_logger(checkpoint_folder+'/test_final_2016.log')
#         trainer.results_folder=checkpoint_folder+'/test_final_2016_npy'
#         trainer.test(last=True)
