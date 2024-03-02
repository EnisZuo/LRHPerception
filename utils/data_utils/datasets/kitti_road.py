import os
import sys
import numpy as np
from typing import Any
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .utils import resize_img, gt_to_binary
sys.path.insert(0, '/home/azuo/LRHPerception')
from configs import cfg
        
class kitti_road_dataset(Dataset):
    def __init__(self, cfgs, split = 'train', img_transform=None, gt_transform=None, save_seg_mask=False):
        self.cfgs = cfgs
        self.save_seg_mask = save_seg_mask
        self.split = split
        if img_transform is not None:
            self.img_transform = img_transform
        if gt_transform is not None:
            self.gt_transform = gt_transform
        data_root = cfgs.DATASET.KITTI_ROOT
        
        if split == 'train':
            self.road_root = os.path.join(data_root, 'data_road/data_road', 'training')
            self.image_root = os.path.join(self.road_root, 'image_2')
            self.gt_root = os.path.join(self.road_root, 'gt_image_2')
            self.image_name_list = os.listdir(self.image_root)
            self.gt_name_list = os.listdir(self.gt_root)
            self.gt_name_list = [each_gt for each_gt in self.gt_name_list if 'road' in each_gt]
            self.sort_image_list()
            self.sort_gt_list()
        elif split == 'test':
            self.road_root = os.path.join(data_root, 'data_road/data_road', 'testing')
            self.image_root = os.path.join(self.road_root, 'image_2')
            self.image_name_list = os.listdir(self.image_root)
            self.sort_image_list()
    
    def sort_image_list(self):
        def sort_key(x):
            prefix, num = x.split('_') 
            return (prefix, int(num.split('.')[0]))
        self.image_name_list.sort(key=sort_key)
        
    def sort_gt_list(self):
        def sort_key(x):
            prefix, num = x.split('_')[0], x.split('_')[2]
            return (prefix, int(num.split('.')[0]))
        self.gt_name_list.sort(key=sort_key)
    
    def __len__(self):
        return len(self.image_name_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root, self.image_name_list[idx])
        img = Image.open(img_path)
        img = self.img_transform(img)
        if(self.split == 'train'):
            gt_img = Image.open(os.path.join(self.gt_root, self.gt_name_list[idx]))
            gt_img = self.gt_transform(gt_img)
            if self.save_seg_mask:
                save_mask = gt_img * 255
                save_mask = Image.fromarray(save_mask.squeeze(0).numpy(), mode='L')
                save_path = os.path.join('/home/azuo/LRHPerception/outputs/road_seg_gt', self.gt_name_list[idx]) + '.png'
                save_mask.save(save_path)
            out = {'img': img, 'img_path': img_path, 'gt_img': gt_img}
            return out
        elif(self.split == 'test'):
            out = {'img': img, 'img_path': img_path}
            return out

def get_kitti_road_dataloader(cfgs, split = 'train', batch_size = 1, save_mask=False):
    if split == 'train':
        resize = cfgs.TRAIN.RESIZE
    else:
        resize = cfgs.EVAL.RESIZE
    
    transform_img = transforms.Compose([
        transforms.ToTensor(),
        resize_img(resize),
    ])
    transform_gt = transforms.Compose([
        transforms.ToTensor(),
        resize_img(resize),
        gt_to_binary(),
    ])
    
    dataset = kitti_road_dataset(cfgs, split = split, img_transform=transform_img, gt_transform=transform_gt, save_seg_mask=save_mask)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    return dataloader
    
if __name__ == '__main__':
    dataloader = get_kitti_road_dataloader(cfg, batch_size=1, save_mask=False)
    # for i, data in enumerate(dataloader):
    #     print(data['gt_img'].shape)

    data = next(iter(dataloader))
    print(data['img'].shape)
    print(data['gt_img'].shape)