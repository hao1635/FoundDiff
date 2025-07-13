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

import ipdb


import matplotlib.pyplot as plt

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list

    return tmplist

# class PDFDataset(Dataset):
#     def __init__(self, phase):
#         #ipdb.set_trace()
#         train_transforms,val_transforms=self.get_transforms()
#         if 'train' in phase:
#             self.transforms = train_transforms
#             num=3000
#         if 'test' in phase:
#             self.transforms = val_transforms
#             num=3000
        
#         self.phase=phase

#         num=3000

#         #ipdb.set_trace()



#         if 'train' in phase:
#             stride = 1
#         if 'test' in phase:
#             stride = 2
#         start=1
#         ab_ndct= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/full_1mm/*')[:num]
#         ab_dose_1_2_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.50/*')[start:num:stride]  # dose 1/2
#         ab_dose_1_3_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.33/*')[start:num:stride]  # dose 1/3
#         ab_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/quarter_1mm/*')[start:num:stride]  # dose 1/4
#         #ab_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.25/*')[:num:stride]  # dose 1/4
#         ab_dose_1_5_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.20/*')[start:num:stride]  # dose 1/5
#         ab_dose_1_6_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.17/*')[start:num:stride]  # dose 1/6
#         ab_dose_1_8_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/sim-0.12/*')[start:num:stride]  # dose 1/8
#         ab_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/sim-0.10/*')[start:num:stride] # dose 1/10
#         ab_dose_1_20_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/sim-0.05/*')[start:num:stride]  # dose 1/20

#         if 'train' in phase:
#             self.ab_images_list=ab_dose_1_2_list+ab_dose_1_4_list+ab_dose_1_6_list+ab_dose_1_10_list+ab_dose_1_3_list[:1500:2]+ab_dose_1_5_list[:1500:2]+ab_dose_1_8_list[:1500:2]+ab_dose_1_20_list[:1500:2]
#         #
#         if 'test' in phase:
#             # self.ab_images_list=ab_dose_1_2_list+ab_dose_1_4_list+ab_dose_1_6_list+ab_dose_1_10_list
#             self.ab_images_list=ab_dose_1_3_list+ab_dose_1_5_list+ab_dose_1_8_list+ab_dose_1_20_list


#         #ipdb.set_trace()
#         if 'train' in phase:
#             stride = 1
#         if 'test' in phase:
#             stride = 2

#         lung_ndct= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/full_1mm/*')[:num]
#         lung_dose_1_2_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.50/*')[start:num:stride]  # dose 1/2
#         lung_dose_1_3_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.33/*')[start:num:stride]  # dose 1/3
#         lung_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.25/*')[start:num:stride]  # dose 1/4
#         lung_dose_1_5_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.20/*')[start:num:stride]  # dose 1/5
#         lung_dose_1_6_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.17/*')[start:num:stride]  # dose 1/6
#         lung_dose_1_8_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/sim-0.12/*')[start:num:stride]  # dose 1/8
#         #lung_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/sim-0.10/*')[:num:stride] # dose 1/10
#         lung_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/quarter_1mm/*')[start:num:stride] # dose 1/10
#         lung_dose_1_20_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/sim-0.05/*')[start:num:stride]  # dose 1/20

#         if 'train' in phase:
#             self.lung_images_list=lung_dose_1_2_list+lung_dose_1_4_list+lung_dose_1_6_list+lung_dose_1_10_list+lung_dose_1_3_list[:1000:4]+lung_dose_1_5_list[:1000:4]+lung_dose_1_8_list[:1000:4]+lung_dose_1_20_list[:1000:4]
#         #self.lung_images_list=lung_dose_1_3_list+lung_dose_1_5_list+lung_dose_1_8_list+lung_dose_1_20_list
#         if 'test' in phase:
#             #self.lung_images_list=lung_dose_1_2_list+lung_dose_1_4_list+lung_dose_1_6_list+lung_dose_1_10_list
#             self.lung_images_list=lung_dose_1_3_list+lung_dose_1_5_list+lung_dose_1_8_list+lung_dose_1_20_list

#         stride=1
#         head_ndct= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+ phase+'/full_1mm/*')[:num]
#         head_dose_1_2_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.50/*')[:num:stride]  # dose 1/2
#         head_dose_1_3_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.33/*')[:num:stride]  # dose 1/3
#         head_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.25/*')[:num:stride]  # dose 1/4
#         #head_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/quarter_1mm/*')[:num:stride]  # dose 1/4
#         head_dose_1_5_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.20/*')[:num:stride]  # dose 1/5
#         head_dose_1_6_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.17/*')[:num:stride]  # dose 1/6
#         head_dose_1_8_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+ phase+'/sim-0.12/*')[:num:stride]  # dose 1/8
#         head_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+ phase+'/sim-0.10/*')[:num:stride] # dose 1/10
#         head_dose_1_20_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+ phase+'/sim-0.05/*')[:num:stride]  # dose 1/20

#         # head_ndct= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/full_1mm/*')[:num]
#         # head_dose_1_2_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.50/*')[:num:stride]  # dose 1/2
#         # head_dose_1_3_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.33/*')[:num:stride]  # dose 1/3
#         # #head_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.25/*')[:num:stride]  # dose 1/4
#         # head_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/quarter_1mm/*')[:num:stride]  # dose 1/4
#         # head_dose_1_5_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.20/*')[:num:stride]  # dose 1/5
#         # head_dose_1_6_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.17/*')[:num:stride]  # dose 1/6
#         # head_dose_1_8_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/sim-0.12/*')[:num:stride]  # dose 1/8
#         # head_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/sim-0.10/*')[:num:stride] # dose 1/10
#         # head_dose_1_20_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/sim-0.05/*')[:num:stride]  # dose 1/20

#         if 'train' in phase:
#             self.head_images_list=head_dose_1_2_list+head_dose_1_4_list+head_dose_1_6_list+head_dose_1_10_list+head_dose_1_3_list[:1000:4]+head_dose_1_5_list[:1000:4]+head_dose_1_8_list[:1000:4]+head_dose_1_20_list[:1000:4]
#         #self.head_images_list=head_dose_1_3_list+head_dose_1_5_list+head_dose_1_8_list+head_dose_1_20_list
#         if 'test' in phase:
#             # self.head_images_list=head_dose_1_2_list+head_dose_1_4_list+head_dose_1_6_list+head_dose_1_10_list
#             self.head_images_list=head_dose_1_3_list+head_dose_1_5_list+head_dose_1_8_list+head_dose_1_20_list

#         # self.q_path_list=sorted_list(opt.dataroot+'/'+phase+'/quarter_1mm/*')
#         # self.f_path_list=sorted_list(opt.dataroot+'/'+phase+'/full_1mm/*')

#         #self.q_path_list= dose_1_2_list+dose_1_3_list+dose_1_4_list+dose_1_5_list+dose_1_6_list+dose_1_8_list+dose_1_10_list+dose_1_20_list
#         self.q_path_list= self.ab_images_list +self.lung_images_list+self.head_images_list
#         # self.f_path_list= ab_ndct

#         self.ab_ndct_path_list= ab_ndct
#         self.lung_ndct_path_list= lung_ndct
#         self.head_ndct_path_list= head_ndct

#         self.A_size = len(self.q_path_list)  # get the size of dataset A
#         # self.B_size = len(self.f_path_list)

#         print('number of images:',self.A_size)
#         self.dataset_size=[len(ab_ndct),len(lung_ndct),len(head_ndct)]




#     def __getitem__(self, index):
        
#         q_data = np.load(self.q_path_list[index]).astype(np.float32)

#         path=self.q_path_list[index]
#         #print(path)
#         loc=path.split('/')[-1].split('-')[0]
#         ndct_index=int(path.split('.')[-2].split('-')[-1])

#         if loc=='head':
#             f_data=np.load(self.head_ndct_path_list[ndct_index]).astype(np.float32)
#             assert self.head_ndct_path_list[ndct_index].split('-')[-1]==self.q_path_list[index].split('-')[-1]
#             B_path=self.head_ndct_path_list[ndct_index]
#         if loc=='lung':
#             f_data=np.load(self.lung_ndct_path_list[ndct_index]).astype(np.float32)
#             assert self.lung_ndct_path_list[ndct_index].split('-')[-1]==self.q_path_list[index].split('-')[-1]
#             B_path=self.lung_ndct_path_list[ndct_index]
#         if loc=='ab':
#             f_data=np.load(self.ab_ndct_path_list[ndct_index]).astype(np.float32)
#             assert self.ab_ndct_path_list[ndct_index].split('-')[-1]==self.q_path_list[index].split('-')[-1]   
#             B_path=self.ab_ndct_path_list[ndct_index]       

#         if self.transforms is not None:
#             q_data = self.transforms[0](q_data)
#             f_data = self.transforms[1](f_data)
        
#         dose=torch.FloatTensor([1.0/self.define_label(path)])
#         feature_vec= dose

#         A = q_data
#         B = f_data
#         A_path=self.q_path_list[index]

#         label=np.asarray(self.define_label(path)).astype(np.float32)
#         #print(B.shape)
        

#         return [B,A]

#     def load_name(self, index, sub_dir=False):
#         #ipdb.set_trace()
#         # condition
#         name = self.q_path_list[index]
#         if sub_dir == 0:
#             return os.path.basename(name)
#         elif sub_dir == 1:
#             path = os.path.dirname(name)
#             sub_dir = (path.split("/"))[-1]
#             return sub_dir+"_"+os.path.basename(name)


#     def define_label(self,path):
#         #print(path)

#         if 'full_1mm' in path:
#             label=1
#             return label
#         if 'quarter_1mm' in path:
#             if 'lung' in path:
#                 label=10
#             else:
#                 label=4
#             return label
#         else:
#             dose=float(path.split('-')[-2])
#             if dose == 0.5:
#                 label=2
#             if dose == 0.33:
#                 label=3
#             if dose == 0.25:
#                 label=4
#             if dose == 0.20:
#                 label=5
#             if dose == 0.17:
#                 label=6
#             if dose == 0.12:
#                 label=8
#             if dose == 0.10:
#                 label=10
#             if dose == 0.05:
#                 label=20

#         return label

#     def __len__(self):
#         """Return the total number of images in the dataset.

#         As we have two datasets with potentially different number of images,
#         we take a maximum of
#         """
#         return self.A_size

#     def get_transforms(self):
#         GLOBAL_RANDOM_STATE = np.random.RandomState(47)
#         seed = GLOBAL_RANDOM_STATE.randint(10000000)
#         RandomState1=np.random.RandomState(seed)
#         RandomState2=np.random.RandomState(seed)

#         min_value = -1000
#         max_value =  2000

#         crop_size=256
#         train_raw_transformer=transforms.Compose([
#         #transforms.CropToFixed(RandomState1, size=(crop_size, crop_size),Center=False),
#         transforms.RandomFlip(RandomState1),
#         transforms.RandomRotate90(RandomState1),
#         transforms.Normalize(min_value=min_value, max_value=max_value),
#         transforms.ToTensor(expand_dims=False)
#         ])

#         train_label_transformer=transforms.Compose([
#         #transforms.CropToFixed(RandomState2, size=(crop_size, crop_size),Center=False),
#         transforms.RandomFlip(RandomState2),
#         transforms.RandomRotate90(RandomState2),
#         transforms.Normalize(min_value=min_value, max_value=max_value),
#         transforms.ToTensor(expand_dims=False)
#         ])

#         val_raw_transformer=transforms.Compose([
#         transforms.Normalize(min_value=min_value, max_value=max_value),
#         transforms.ToTensor(expand_dims=False)
#         ])

#         val_label_transformer=transforms.Compose([
#         transforms.Normalize(min_value=min_value, max_value=max_value),
#         transforms.ToTensor(expand_dims=False)
#         ])

#         train_transforms=[train_raw_transformer,train_label_transformer]
#         val_transforms=[val_raw_transformer,val_label_transformer]

#         return train_transforms,val_transforms

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

import ipdb


import matplotlib.pyplot as plt

def sorted_list(path): 
    tmplist = glob.glob(path) # finding all files or directories and listing them.
    tmplist.sort() # sorting the found list

    return tmplist

class PDFDataset(Dataset):
    def __init__(self, phase,mode='2020_seen'):
        #ipdb.set_trace()
        train_transforms,val_transforms=self.get_transforms()
        if 'train' in phase:
            self.transforms = train_transforms
            num=3000
        if 'test' in phase:
            self.transforms = val_transforms
            num=3000
        
        self.phase=phase

        num=3000

        #ipdb.set_trace()



        if 'train' in phase:
            stride = 2
        if 'test' in phase:
            stride = 2
        start=0
        ab_ndct= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/full_1mm/*')[:num]
        ab_dose_1_2_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.50/*')[start:num:stride]  # dose 1/2
        ab_dose_1_3_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.33/*')[start:num:stride]  # dose 1/3
        ab_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/quarter_1mm/*')[start:num:stride]  # dose 1/4
        #ab_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.25/*')[:num:stride]  # dose 1/4
        ab_dose_1_5_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.20/*')[start:num:stride]  # dose 1/5
        ab_dose_1_6_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+phase+'/sim-0.17/*')[start:num:stride]  # dose 1/6
        ab_dose_1_8_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/sim-0.12/*')[start:num:stride]  # dose 1/8
        ab_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/sim-0.10/*')[start:num:stride] # dose 1/10
        ab_dose_1_20_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_ab_2d/'+ phase+'/sim-0.05/*')[start:num:stride]  # dose 1/20


        # self.ab_images_list= ab_dose_1_10_list
        self.ab_images_list=ab_dose_1_2_list+ab_dose_1_4_list+ab_dose_1_6_list+ab_dose_1_10_list
        # self.ab_images_list=ab_dose_1_3_list+ab_dose_1_5_list+ab_dose_1_8_list+ab_dose_1_20_list


        #ipdb.set_trace()
        if 'train' in phase:
            stride = 2
        if 'test' in phase:
            stride = 2

        lung_ndct= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/full_1mm/*')[:num]
        lung_dose_1_2_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.50/*')[start:num:stride]  # dose 1/2
        lung_dose_1_3_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.33/*')[start:num:stride]  # dose 1/3
        lung_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.25/*')[start:num:stride]  # dose 1/4
        lung_dose_1_5_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.20/*')[start:num:stride]  # dose 1/5
        lung_dose_1_6_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+phase+'/sim-0.17/*')[start:num:stride]  # dose 1/6
        lung_dose_1_8_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/sim-0.12/*')[start:num:stride]  # dose 1/8
        #lung_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/sim-0.10/*')[:num:stride] # dose 1/10
        lung_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/quarter_1mm/*')[start:num:stride] # dose 1/10
        lung_dose_1_20_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_lung_2d/'+ phase+'/sim-0.05/*')[start:num:stride]  # dose 1/20


        # self.lung_images_list=lung_dose_1_10_list
        self.lung_images_list=lung_dose_1_2_list+lung_dose_1_4_list+lung_dose_1_6_list+lung_dose_1_10_list
        # self.lung_images_list=lung_dose_1_3_list+lung_dose_1_5_list+lung_dose_1_8_list+lung_dose_1_20_list

        stride=1
        # head_ndct= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+ phase+'/full_1mm/*')[:num]
        # head_dose_1_2_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.50/*')[:num:stride]  # dose 1/2
        # head_dose_1_3_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.33/*')[:num:stride]  # dose 1/3
        # head_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.25/*')[:num:stride]  # dose 1/4
        # #head_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/quarter_1mm/*')[:num:stride]  # dose 1/4
        # head_dose_1_5_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.20/*')[:num:stride]  # dose 1/5
        # head_dose_1_6_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+phase+'/sim-0.17/*')[:num:stride]  # dose 1/6
        # head_dose_1_8_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+ phase+'/sim-0.12/*')[:num:stride]  # dose 1/8
        # head_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+ phase+'/sim-0.10/*')[:num:stride] # dose 1/10
        # head_dose_1_20_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d/'+ phase+'/sim-0.05/*')[:num:stride]  # dose 1/20

        head_ndct= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/full_1mm/*')[:num]
        head_dose_1_2_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.50/*')[:num:stride]  # dose 1/2
        head_dose_1_3_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.33/*')[:num:stride]  # dose 1/3
        #head_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.25/*')[:num:stride]  # dose 1/4
        head_dose_1_4_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/quarter_1mm/*')[:num:stride]  # dose 1/4
        head_dose_1_5_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.20/*')[:num:stride]  # dose 1/5
        head_dose_1_6_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+phase+'/sim-0.17/*')[:num:stride]  # dose 1/6
        head_dose_1_8_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/sim-0.12/*')[:num:stride]  # dose 1/8
        head_dose_1_10_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/sim-0.10/*')[:num:stride] # dose 1/10
        head_dose_1_20_list= sorted_list('/mnt/miah203/zhchen/Mayo2020_head_2d_2/'+ phase+'/sim-0.05/*')[:num:stride]  # dose 1/20


        # self.head_images_list=head_dose_1_10_list
        self.head_images_list=head_dose_1_2_list+head_dose_1_4_list+head_dose_1_6_list+head_dose_1_10_list
        # self.head_images_list=head_dose_1_3_list+head_dose_1_5_list+head_dose_1_8_list+head_dose_1_20_list

        self.mayo16_ldct=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/quarter_1mm/*')
        self.mayo16_ndct_path_list=sorted_list('/mnt/miah203/zhchen/Mayo2016_2d/train/full_1mm/*')

        

        #self.q_path_list= dose_1_2_list+dose_1_3_list+dose_1_4_list+dose_1_5_list+dose_1_6_list+dose_1_8_list+dose_1_10_list+dose_1_20_list
        # self.q_path_list= self.ab_images_list +self.lung_images_list+self.head_images_list #+self.mayo16_ldct
        # self.f_path_list= ab_ndct
        self.q_path_list=head_dose_1_10_list
        # self.q_path_list = self.q_path_list[:50]
        if 'test' in phase:
            self.q_path_list=self.q_path_list

        self.ab_ndct_path_list= ab_ndct
        self.lung_ndct_path_list= lung_ndct
        self.head_ndct_path_list= head_ndct

        self.A_size = len(self.q_path_list)  # get the size of dataset A
        # self.B_size = len(self.f_path_list)

        print('number of images:',self.A_size)
        self.dataset_size=[len(ab_ndct),len(lung_ndct),len(head_ndct)]

        


    def __getitem__(self, index):
        
        q_data = np.load(self.q_path_list[index]).astype(np.float32)

        path=self.q_path_list[index]
        #print(path)
        loc=path.split('/')[-1].split('-')[0]
        ndct_index=int(path.split('.')[-2].split('-')[-1])

        if loc=='head':
            f_data=np.load(self.head_ndct_path_list[ndct_index]).astype(np.float32)
            assert self.head_ndct_path_list[ndct_index].split('-')[-1]==self.q_path_list[index].split('-')[-1]
            B_path=self.head_ndct_path_list[ndct_index]
        if loc=='lung':
            f_data=np.load(self.lung_ndct_path_list[ndct_index]).astype(np.float32)
            assert self.lung_ndct_path_list[ndct_index].split('-')[-1]==self.q_path_list[index].split('-')[-1]
            B_path=self.lung_ndct_path_list[ndct_index]
        if loc=='ab':
            f_data=np.load(self.ab_ndct_path_list[ndct_index]).astype(np.float32)
            assert self.ab_ndct_path_list[ndct_index].split('-')[-1]==self.q_path_list[index].split('-')[-1]   
            B_path=self.ab_ndct_path_list[ndct_index]  

        if  loc=='mayo16':  
            f_data=np.load(self.mayo16_ndct_path_list[ndct_index]).astype(np.float32)
            assert self.ab_ndct_path_list[ndct_index].split('-')[-1]==self.q_path_list[index].split('-')[-1]   
            B_path=self.ab_ndct_path_list[ndct_index]  

        if self.transforms is not None:
            q_data = self.transforms[0](q_data)
            f_data = self.transforms[1](f_data)
        
        dose=torch.FloatTensor([1.0/self.define_label(path)])
        feature_vec= dose

        A = q_data
        B = f_data
        A_path=self.q_path_list[index]

        label=np.asarray(self.define_label(path)).astype(np.float32)
        #print(B.shape)
        

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


    def define_label(self,path):
        #print(path)

        if 'full_1mm' in path:
            label=1
            return label
        if 'quarter_1mm' in path:
            if 'lung' in path:
                label=10
            else:
                label=4
            return label
        else:
            dose=float(path.split('-')[-2])
            if dose == 0.5:
                label=2
            if dose == 0.33:
                label=3
            if dose == 0.25:
                label=4
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

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.A_size

    def get_transforms(self):
        GLOBAL_RANDOM_STATE = np.random.RandomState(47)
        seed = GLOBAL_RANDOM_STATE.randint(10000000)
        RandomState1=np.random.RandomState(seed)
        RandomState2=np.random.RandomState(seed)

        min_value = -1000
        max_value =  2000

        crop_size=256
        train_raw_transformer=transforms.Compose([
        #transforms.CropToFixed(RandomState1, size=(crop_size, crop_size),Center=False),
        transforms.RandomFlip(RandomState1),
        transforms.RandomRotate90(RandomState1),
        transforms.Normalize(min_value=min_value, max_value=max_value),
        transforms.ToTensor(expand_dims=False)
        ])

        train_label_transformer=transforms.Compose([
        #transforms.CropToFixed(RandomState2, size=(crop_size, crop_size),Center=False),
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







