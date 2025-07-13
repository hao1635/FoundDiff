# FoundDiff
Official implementation of "FoundDiff: Foundational Diffusion Model for Generalizable Low-Dose CT Denoising" 


## Updates
July, 2025: initial commit.  


## Data Preparation
The 2016 AAPM-Mayo dataset can be downloaded from: [CT Clinical Innovation Center](https://ctcicblog.mayo.edu/2016-low-dose-ct-grand-challenge/) (B30 kernel)  
The 2020 AAPM-Mayo dataset can be downloaded from: [cancer imaging archive](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026)   


## Requirements
```
- Linux Platform
- torch==1.12.1+cu113 # depends on the CUDA version of your machine
- torchvision==0.13.1+cu113
- Python==3.8.0
- numpy==1.22.3
```

## Traning and & Inference


#### Training:  
```
CUDA_VISIBLE_DEVICES=1 python train.py --name FoundDiff --is_train --train_num_steps 400000
```

#### Inference & testing:
```
CUDA_VISIBLE_DEVICES=4 python train.py --name FoundDiff --epoch 300 --dataset 2020_seen --eval
```
Please refer to options files for more setting.


