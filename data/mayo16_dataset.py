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
from . import transforms
from data.base_dataset import BaseDataset, get_transform
from PIL import Image
from torch.utils.data import Dataset


import matplotlib.pyplot as plt

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list

    return tmplist

class Mayo16Dataset(Dataset):
    def __init__(self,phase):
        #ipdb.set_trace()
        train_transforms,val_transforms=self.get_transforms()
        if 'train' in phase:
            self.transforms = train_transforms
        if 'test' in phase:
            self.transforms = val_transforms
        
        self.phase=phase


        if 'train' in phase:
            #self.q_path_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/train/quarter_1mm/*') #[0::2]
            #self.f_path_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/train/full_1mm/*') #[0::2]
            #self.q_path_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/test/sim-0.25/*') #[0::2]
            self.f_path_list = sorted_list('/mnt/miah203/zhchen/CQ500_2d/test/full_1mm/*')
            self.q_path_list = sorted_list('/mnt/miah203/zhchen/CQ500_2d/test/sim-0.25/*')

        if 'test' in phase:
            self.q_path_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/test/quarter_1mm/*')
            # self.q_path_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/test/sim-0.25/*')
            self.f_path_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/test/full_1mm/*')
            # self.f_path_list = sorted_list('/mnt/miah203/zhchen/CQ500_2d/test/full_1mm/*')
            # self.q_path_list = sorted_list('/mnt/miah203/zhchen/CQ500_2d/test/sim-0.25/*')
            #self.q_path_list=sorted_list('/mnt/miah203/zhchen/肺科医院/ldct_npy_2d/*')[:2000:2]
            #self.f_path_list=sorted_list('/mnt/miah203/zhchen/肺科医院/ldct_npy_2d/*')[:2000:2]

        self.A_size = len(self.q_path_list)  # get the size of dataset A
        self.B_size = len(self.f_path_list)


        self.A_size = len(self.q_path_list)  # get the size of dataset A
        self.B_size = len(self.f_path_list)



    def __getitem__(self, index):
        #assert self.f_path_list[index].split('-')[-1]==self.q_path_list[index].split('-')[-1]

        f_data=np.load(self.f_path_list[index]).astype(np.float32)
        q_data = np.load(self.q_path_list[index]).astype(np.float32)


        if self.transforms is not None:
            q_data = self.transforms[0](q_data)
            f_data = self.transforms[1](f_data)
        
        #weights=self._get_weights(f_data)
        weights=0

        A = q_data
        B = f_data
        A_path=self.q_path_list[index]
        B_path=self.f_path_list[index]

        return [B,A]

    def load_name(self, index, sub_dir=False):
        #ipdb.set_trace()
        # condition
        name = self.q_path_list[index]
        if sub_dir == 0:
            return os.path.basename(name)
        elif sub_dir == 1:
            path = os.path.dirname(name)
            sub_dir = (path.split("/"))[-1]
            return sub_dir+"_"+os.path.basename(name)

# class Mayo16Dataset(BaseDataset):
#     def __init__(self, opt):
#         #ipdb.set_trace()
#         train_transforms,val_transforms=self.get_transforms(opt)
#         if 'train' in opt.phase:
#             self.transforms = train_transforms
#         if 'test' in opt.phase:
#             self.transforms = val_transforms
        
#         self.phase=opt.phase
#         self.opt=opt

#         self.ab_LDCT_list=sorted_list('/data/zhchen/Mayo2020_ab_2d/'+opt.phase+'/quarter_1mm/*')
#         self.chest_LDCT_list=sorted_list('/data/zhchen/Mayo2020_lung_2d/'+opt.phase+'/quarter_1mm/*')
#         self.brain_LDCT_list=sorted_list('/data/zhchen/Mayo2020_head_2d/'+opt.phase+'/quarter_1mm/*')

#         self.ab_NDCT_list=sorted_list('/data/zhchen/Mayo2020_ab_2d/'+opt.phase+'/full_1mm/*')
#         self.chest_NDCT_list=sorted_list('/data/zhchen/Mayo2020_lung_2d/'+opt.phase+'/full_1mm/*')
#         self.brain_NDCT_list=sorted_list('/data/zhchen/Mayo2020_head_2d/'+opt.phase+'/full_1mm/*')

#         # self.q_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/quarter_1mm/*')
#         # self.f_path_list=sorted_list(opt.dataroot+'/'+opt.phase+'/full_1mm/*')

#         self.q_path_list=self.ab_LDCT_list+self.chest_LDCT_list+self.brain_LDCT_list
#         self.f_path_list= self.ab_NDCT_list+self.chest_NDCT_list+self.brain_NDCT_list

#         self.A_size = len(self.q_path_list)  # get the size of dataset A
#         self.B_size = len(self.f_path_list)



#     def __getitem__(self, index):
#         assert self.f_path_list[index].split('-')[-1]==self.q_path_list[index].split('-')[-1]

#         f_data=np.load(self.f_path_list[index]).astype(np.float32)
#         q_data = np.load(self.q_path_list[index]).astype(np.float32)


#         if self.transforms is not None:
#             q_data = self.transforms[0](q_data)
#             f_data = self.transforms[1](f_data)
        
#         #weights=self._get_weights(f_data)
#         weights=0

#         A = q_data
#         B = f_data
#         A_path=self.q_path_list[index]
#         B_path=self.f_path_list[index]

#         return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path,'weights':weights}



    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def get_transforms(self):
        GLOBAL_RANDOM_STATE = np.random.RandomState(47)
        seed = GLOBAL_RANDOM_STATE.randint(10000000)
        RandomState1=np.random.RandomState(seed)
        RandomState2=np.random.RandomState(seed)

        min_value = -1000
        max_value =  2000

        train_raw_transformer=transforms.Compose([
        #transforms.CropToFixed(RandomState1, size=(opt.crop_size, opt.crop_size)),
        transforms.RandomFlip(RandomState1),
        transforms.RandomRotate90(RandomState1),
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        train_label_transformer=transforms.Compose([
        #transforms.CropToFixed(RandomState2, size=(opt.crop_size, opt.crop_size)),
        transforms.RandomFlip(RandomState2),
        transforms.RandomRotate90(RandomState2),
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        val_raw_transformer=transforms.Compose([
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        val_label_transformer=transforms.Compose([
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        train_transforms=[train_raw_transformer,train_label_transformer]
        val_transforms=[val_raw_transformer,val_label_transformer]

        return train_transforms,val_transforms



