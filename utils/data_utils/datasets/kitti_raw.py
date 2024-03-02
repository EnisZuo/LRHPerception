import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from .utils import resize_img
from torchvision import transforms
sys.path.insert(0, '/home/azuo/LRHPerception')
from configs import cfg

class kitti_raw_dataset(Dataset):
    def __init__(self, cfgs, transform=None):
        self.root_dir = cfgs.DEPTH.DATA_PATH
        self.sorted_filenames_file =  cfgs.DEPTH.FILENAMES_FILE
        self.filename_list = []
        self.transform = transform
        with open(self.sorted_filenames_file, 'r') as f:
            content = f.read()
            list_content = content.split()
        self.filename_list = [k for k in list_content if 'png' in k]
        self.filename_list = sorted(self.filename_list)

    def __len__(self):
        return len(self.filename_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filename_list[idx].replace('\n', ''))
        img = Image.open(img_path)
        img = self.transform(img)
        out = {'img': img, 'img_path': img_path}
        return out
        

def get_kitti_raw_dataloader(cfgs, split='test', batch_size=1, transform=None):
    if split == 'test':
        resize = cfgs.EVAL.RESIZE
    transform = transforms.Compose([
        transforms.ToTensor(),
        resize_img(resize)
    ])
    
    dataset = kitti_raw_dataset(cfgs.DATASET, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    return dataloader
    
if __name__ == '__main__':
    dataloader = get_kitti_raw_dataloader(cfg)
    data = next(iter(dataloader))
    img = data['img']
    print(img.shape)