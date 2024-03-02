import os
import sys
import cv2
import torch
import torch.nn.functional as F
from .utils import resize_img
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '/home/azuo/LRHPerception')
from configs import cfg

class kitti_det_dataset(Dataset):
    def __init__(self, cfg, split, transform=None):
        self.cfg = cfg
        self.split = split
        self.det_root = self.cfg.DATASET.DETECT.DET_ROOT
        self.transform = transform
        self.class_to_idx = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Cyclist': 4}
        
        def sort_key(x):
            return int(x.split('.')[0])
        
        if self.split == 'train':
            train_root = os.path.join(self.det_root, 'training')
            self.img_root = os.path.join(train_root, 'image_2')
            self.label_root = os.path.join(train_root, 'label_2')
            self.img_list = os.listdir(self.img_root)
            self.img_list.sort(key=sort_key)
            self.resize = cfg.TRAIN.RESIZE
        elif self.split == 'test':
            train_root = os.path.join(self.det_root, 'training')
            self.img_root = os.path.join(train_root, 'image_2')
            self.img_list = os.listdir(self.img_root)
            self.resize = cfg.TRAIN.RESIZE # for now
    def __len__(self):
        return len(self.img_list)
            
    def __getitem__(self, index):
        if self.split == 'train':
            img_name = self.img_list[index]
            img_path = os.path.join(self.img_root, img_name)
            label_path = os.path.join(self.label_root, img_name.replace('png', 'txt'))
            labels = []
            img = Image.open(img_path)
            resize_h, resize_w = self.resize
            img_w, img_h = img.size
            if self.transform is None:
                img = transforms.ToTensor()(img)
            else:
                img = self.transform(img)
            
            with open(label_path) as f:
                labels = f.readlines()
            labels = [label.split() for label in labels]
            converted_labels = []
            for label in labels:
                try:
                    class_idx = self.class_to_idx[label[0]]
                except KeyError:
                    continue
                tl_x = float(label[4])
                tl_y = float(label[5])
                br_x = float(label[6])
                br_y = float(label[7])
                cx = resize_w * ((tl_x + br_x) / 2) / img_w
                cy = resize_h * ((tl_y + br_y) / 2) / img_h
                width = resize_w * (br_x - tl_x) / img_w
                height = resize_h * (br_y - tl_y) / img_h
                converted_labels.append([class_idx, cx, cy, width, height])
            converted_labels = torch.tensor(converted_labels, dtype=torch.float32)
            out = {'img': img, 'img_path': img_path, 'label': converted_labels}
            return out
        
        elif self.split == 'test':
            img_name = self.img_list[index]
            img_path = os.path.join(self.img_root, img_name)
            img = Image.open(img_path)
            if self.transform is None:
                img = transforms.ToTensor()(img)
            else:
                img = self.transform(img)
            out = {'img': img, 'img_path': img_path, 'label': None}
            return out

def get_kitti_det_dataloader(cfgs, split, batch_size=1):
    class NewDataLoader(object):
        def __init__(self, cfgs, split, batch_size):
            if split == 'train':
                resize = cfgs.TRAIN.RESIZE
            img_transform = transforms.Compose([
                transforms.ToTensor(),
                resize_img(resize)
            ])
            dataset = kitti_det_dataset(cfgs, split, transform=img_transform)

            if cfgs.DISTRIBUTED:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                self.train_sampler = None
            self.data = DataLoader(dataset, shuffle=(self.train_sampler is None), batch_size=batch_size, collate_fn=custom_collate,\
                                    num_workers=cfgs.NUM_THREADS, pin_memory=True)
    return NewDataLoader(cfgs, split, batch_size)

def custom_collate(batch):
    imgs = [item['img'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    labels = [item['label'] for item in batch]
    max_num_objects = max(label.shape[0] for label in labels)
    padded_labels = []
    for label in labels:
        pad_num = max_num_objects - label.shape[0]
        padded_label = F.pad(label, (0, 0, 0, pad_num), value=-1)
        padded_labels.append(padded_label)

    return torch.stack(imgs), img_paths, torch.stack(padded_labels)
    
if __name__ == '__main__':
    dataloader = get_kitti_det_dataloader(cfg=cfg, split='train')
    data = next(iter(dataloader))