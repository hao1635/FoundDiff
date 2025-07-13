#dataset
import os, glob, shutil
import numpy as np
import tqdm
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
import ipdb
from util import transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image


import matplotlib.pyplot as plt

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list

    return tmplist

class DoseDataset(BaseDataset):
    def __init__(self, opt):
        #ipdb.set_trace()
        train_transforms,val_transforms=self.get_transforms(opt)

        if 'train' in opt.phase:
            self.transforms = train_transforms
        if 'test' in opt.phase:
            self.transforms = val_transforms
        
        self.phase=opt.phase
        self.opt=opt 

        num=5000

        ab_ndct= sorted_list('/data/zhchen/Mayo2020_ab_2d/'+ opt.phase+'/full_1mm/*')[:num]
        ab_dose_1_2_list= sorted_list('/data/zhchen/Mayo2020_ab_2d/'+opt.phase+'/sim-0.50/*')[:num]  # dose 1/2
        ab_dose_1_3_list= sorted_list('/data/zhchen/Mayo2020_ab_2d/'+opt.phase+'/sim-0.33/*')[:num]  # dose 1/3
        ab_dose_1_4_list= sorted_list('//data/zhchen/Mayo2020_ab_2d/'+opt.phase+'/sim-0.25/*')[:num]  # dose 1/4
        ab_dose_1_5_list= sorted_list('/data/zhchen/Mayo2020_ab_2d/'+opt.phase+'/sim-0.20/*')[:num]  # dose 1/5
        ab_dose_1_6_list= sorted_list('/data/zhchen/Mayo2020_ab_2d/'+opt.phase+'/sim-0.17/*')[:num]  # dose 1/6
        ab_dose_1_8_list= sorted_list('/data/zhchen/Mayo2020_ab_2d/'+ opt.phase+'/sim-0.12/*')[:num]  # dose 1/8
        ab_dose_1_10_list= sorted_list('/data/zhchen/Mayo2020_ab_2d/'+ opt.phase+'/sim-0.10/*')[:num] # dose 1/10
        ab_dose_1_20_list= sorted_list('/data/zhchen/Mayo2020_ab_2d/'+ opt.phase+'/sim-0.05/*')[:num]  # dose 1/20

        self.ab_images_list=ab_ndct+ab_dose_1_2_list+ab_dose_1_3_list+ab_dose_1_4_list+ab_dose_1_5_list+ab_dose_1_6_list+ab_dose_1_8_list+ab_dose_1_10_list+ab_dose_1_20_list

        lung_ndct= sorted_list('/data/zhchen/Mayo2020_lung_2d/'+ opt.phase+'/full_1mm/*')[:num]
        lung_dose_1_2_list= sorted_list('/data/zhchen/Mayo2020_lung_2d/'+opt.phase+'/sim-0.50/*')[:num]  # dose 1/2
        lung_dose_1_3_list= sorted_list('/data/zhchen/Mayo2020_lung_2d/'+opt.phase+'/sim-0.33/*')[:num]  # dose 1/3
        lung_dose_1_4_list= sorted_list('//data/zhchen/Mayo2020_lung_2d/'+opt.phase+'/sim-0.25/*')[:num]  # dose 1/4
        lung_dose_1_5_list= sorted_list('/data/zhchen/Mayo2020_lung_2d/'+opt.phase+'/sim-0.20/*')[:num]  # dose 1/5
        lung_dose_1_6_list= sorted_list('/data/zhchen/Mayo2020_lung_2d/'+opt.phase+'/sim-0.17/*')[:num]  # dose 1/6
        lung_dose_1_8_list= sorted_list('/data/zhchen/Mayo2020_lung_2d/'+ opt.phase+'/sim-0.12/*')[:num]  # dose 1/8
        lung_dose_1_10_list= sorted_list('/data/zhchen/Mayo2020_lung_2d/'+ opt.phase+'/sim-0.10/*')[:num] # dose 1/10
        lung_dose_1_20_list= sorted_list('/data/zhchen/Mayo2020_lung_2d/'+ opt.phase+'/sim-0.05/*')[:num]  # dose 1/20

        self.lung_images_list=lung_ndct+lung_dose_1_2_list+lung_dose_1_3_list+lung_dose_1_4_list+lung_dose_1_5_list+lung_dose_1_6_list+lung_dose_1_8_list+lung_dose_1_10_list+lung_dose_1_20_list

        head_ndct= sorted_list('/data/zhchen/Mayo2020_head_2d/'+ opt.phase+'/full_1mm/*')[:num]
        head_dose_1_2_list= sorted_list('/data/zhchen/Mayo2020_head_2d/'+opt.phase+'/sim-0.50/*')[:num]  # dose 1/2
        head_dose_1_3_list= sorted_list('/data/zhchen/Mayo2020_head_2d/'+opt.phase+'/sim-0.33/*')[:num]  # dose 1/3
        head_dose_1_4_list= sorted_list('//data/zhchen/Mayo2020_head_2d/'+opt.phase+'/sim-0.25/*')[:num]  # dose 1/4
        head_dose_1_5_list= sorted_list('/data/zhchen/Mayo2020_head_2d/'+opt.phase+'/sim-0.20/*')[:num]  # dose 1/5
        head_dose_1_6_list= sorted_list('/data/zhchen/Mayo2020_head_2d/'+opt.phase+'/sim-0.17/*')[:num]  # dose 1/6
        head_dose_1_8_list= sorted_list('/data/zhchen/Mayo2020_head_2d/'+ opt.phase+'/sim-0.12/*')[:num]  # dose 1/8
        head_dose_1_10_list= sorted_list('/data/zhchen/Mayo2020_head_2d/'+ opt.phase+'/sim-0.10/*')[:num] # dose 1/10
        head_dose_1_20_list= sorted_list('/data/zhchen/Mayo2020_head_2d/'+ opt.phase+'/sim-0.05/*')[:num]  # dose 1/20

        self.head_images_list=head_ndct+head_dose_1_2_list+head_dose_1_3_list+head_dose_1_4_list+head_dose_1_5_list+head_dose_1_6_list+head_dose_1_8_list+head_dose_1_10_list+head_dose_1_20_list

        self.images_list=self.ab_images_list #+self.lung_images_list+self.head_images_list



    def __getitem__(self, index):
        
        path=self.images_list[index]
        image_data=np.load(self.images_list[index]).astype(np.float32)

        if self.transforms is not None:
            img = [self.transforms(image_data),self.transforms(image_data)]

        label=np.asarray(self.define_label(path)).astype(np.float32)

        return img, label
        



    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.images_list)

    def define_label(self,path):
        #print(path)

        if 'full_1mm' in path:
            label=1
            return label
        if 'quarter_1mm' in path:
            label=4
            return label
        else:
            dose=float(path.split('-')[-2])
            if dose == 0.5:
                label=2
            if dose == 0.33:
                label=3
            if dose == 0.20:
                label=5
            if dose == 0.17:
                label=6
            if dose == 0.12:
                label=8
            if dose == 0.10:
                label=10
            if dose == 0.05:
                label=20

        return label


    def get_transforms(self,opt):
        GLOBAL_RANDOM_STATE = np.random.RandomState(47)
        seed = GLOBAL_RANDOM_STATE.randint(10000000)
        RandomState1=np.random.RandomState(seed)
        RandomState2=np.random.RandomState(seed)

        min_value = -1000
        max_value =  2000

        train_transformer=transforms.Compose([
        transforms.CropToFixed(RandomState1, size=(opt.crop_size, opt.crop_size),centered=True),
        transforms.RandomFlip(RandomState1),
        transforms.RandomRotate90(RandomState1),
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])


        val_transformer=transforms.Compose([
        #transforms.CropToFixed(RandomState1, size=(opt.crop_size, opt.crop_size),centered=True),
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])



        return train_transformer,val_transformer



