import cv2
import os, sys, torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .utils import resize_img
sys.path.insert(0, '/home/azuo/LRHPerception')
from configs import cfg

class kitti_seg_dataset(Dataset):
    def __init__(self, cfgs, split = 'train'):
        self.data_root = cfgs.DATASET.KITTI_SEG.ROOT
        self.split = split
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            resize_img(cfgs.TRAIN.RESIZE)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            resize_img(cfgs.TRAIN.RESIZE)
        ])
        
        if split == 'train':
            self.img_root = os.path.join(self.data_root, 'training', 'image_2')
            self.gt_root = os.path.join(self.data_root, 'training', 'road_gt')
            self.img_list = os.listdir(self.img_root)
            
            
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_root, img_name)
        img = cv2.imread(img_path)
        img = self.img_transform(img)
        
        if self.split == 'train':
            gt_path = os.path.join(self.gt_root, img_name)
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            gt = self.gt_transform(gt)
            
            out = {'img': img, 'img_path': img_path, 'gt_img': gt}
        return out

def get_kitti_seg_dataloader(cfgs, split = 'train', batch_size = 1):
    class NewDataloader(object):
        def __init__(self, cfgs, split, batch_size):
            dataset = kitti_seg_dataset(cfgs, split)
            if cfgs.DISTRIBUTED:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                train_sampler = None
            self.data = DataLoader(dataset, batch_size=batch_size, shuffle=(train_sampler is None), pin_memory=True)
    return NewDataloader(cfgs, split, batch_size)
    
if __name__ == '__main__':
    dataloader = get_kitti_seg_dataloader(cfg)
    for iter, data in enumerate(dataloader):
        print(data['gt_img'].shape)