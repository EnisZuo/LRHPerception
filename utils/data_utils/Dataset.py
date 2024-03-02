import os, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class CustomDataset(Dataset):
    def __init__(self, cfgs, root, split='train', transform=None, batch_size=16):
        self.batch_size = batch_size
        self.root = root
        self.split = split
        self.transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor()
    ])
        self.index = 0
        self.resize = cfgs.TRAIN.RESIZE

        self.image_folder = os.path.join(root, 'images', split)
        self.detect_labels_folder = os.path.join(root, 'detect_labels', split)
        self.seg_labels_folder = os.path.join(root, 'seg_labels', split)

        self.image_filenames = [f for f in os.listdir(self.detect_labels_folder) if f.endswith('.txt')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        resize_h, resize_w = self.resize
        self.index = 0 if self.index == self.batch_size else self.index
        img_name = self.image_filenames[idx].replace('.txt', '.png')
        img_path = os.path.join(self.image_folder, img_name)
        detect_label_path = os.path.join(self.detect_labels_folder, self.image_filenames[idx])
        seg_label_path = os.path.join(self.seg_labels_folder, self.image_filenames[idx])

        image = Image.open(img_path)
        seg_label = np.loadtxt(seg_label_path)
        img_w, img_h = image.size

        # if os.stat(detect_label_path).st_size == 0:         #Fill zeros if label is empty
        #     # batch_idx = torch.tensor([self.index], dtype=torch.float)
        #     # cls = torch.zeros((1), dtype=torch.float)
        detect_label = torch.zeros((0, 5), dtype=torch.float)
        # else:
        #     detect_label = np.loadtxt(detect_label_path)
        #     if len(detect_label.shape) == 1:
        #         detect_label = detect_label.reshape(1, -1)

        detect_label = torch.tensor(detect_label, dtype=torch.float32)
        # seg_label = 
        if self.transform:
            image = self.transform(image)

        sample = {
            'image': image,
            'detect_label': detect_label,
            'seg_label': torch.tensor(seg_label, dtype=torch.float)
        }
        self.index += 1
        return sample
    
class Dataset_MOT(Dataset):
        def __init__(self, mot_root, yolo_annotation_root, split='train', transform=None, batch_size=1):
            self.batch_size = batch_size
            if(split == 'test'):
                self.mot_root = os.path.join(mot_root, split) # e.g. 'data/MOT17_coco'
            else:
                self.mot_root = os.path.join(mot_root, 'train') # e.g. 'data
            self.yolo_annotation_root = os.path.join(yolo_annotation_root, split) # e.g. 'data/MOT17_yolo'
            self.split = split
            self.transform = transform
            self.index = 0


            self.detect_labels_folder = os.path.join(yolo_annotation_root, 'detect_labels', split)
            # self.seg_labels_folder = os.path.join(root, 'seg_labels', split)

            self.image_filenames = [f for f in os.listdir(self.detect_labels_folder) if f.endswith('.txt')]
            self.image_filenames = sorted(self.image_filenames)
            # print(self.image_filenames)
        
        def __len__(self):
            return len(self.image_filenames)
        
        def __getitem__(self, idx):
            self.index = 0 if self.index == self.batch_size else self.index
            # print('self.index: ', self.index)
            print(idx)
            img_rel_path = self.image_filenames[idx].replace('.txt', '.jpg').replace('_', '/')
            img_path = os.path.join(self.mot_root, img_rel_path)
            # print('img_rel_path from mot_dataloader: ', img_rel_path)
            # print("img_path from mot_dataloader: ", img_path)
            
            detect_label_path = os.path.join(self.detect_labels_folder, self.image_filenames[idx])
            
            # seg_label_path = os.path.join(self.seg_labels_folder, self.image_filenames[idx])
            seg_label_path = '/home/azuo/Trajectory_Prediction/DSPNet/data/seg_labels/val/frankfurt_000000_000294_leftImg8bit.txt'
            
            image = Image.open(img_path)
            seg_label = np.loadtxt(seg_label_path)

            if os.stat(detect_label_path).st_size == 0:         #Fill zeros if label is empty
                batch_idx = torch.tensor([self.index], dtype=torch.float)
                cls = torch.zeros((1), dtype=torch.float)
                bboxes = torch.zeros((0, 4), dtype=torch.float)
            else:
                detect_label = np.loadtxt(detect_label_path)
                if len(detect_label.shape) == 1:
                    detect_label = detect_label.reshape(1, -1)
                batch_idx = torch.full((detect_label.shape[0], 1), self.index, dtype=torch.float)
                cls = torch.tensor(detect_label[:, 0], dtype=torch.float)
                bboxes = torch.tensor(detect_label[:, 1:], dtype=torch.float)

            if self.transform:
                image = self.transform(image)

            sample = {
                'image': image,
                'detect_label': {'batch_idx': batch_idx, 'cls': cls, 'bboxes': bboxes},
                'img_rel_path': img_rel_path,
                'img_path': img_path
                # 'seg_label': torch.tensor(seg_label, dtype=torch.float)
            }
            # print(detect_label)

            self.index += 1
            return sample
        
        
def custom_collate(batch):
    imgs = [item['image'] for item in batch]
    labels = [item['detect_label'] for item in batch]
    seg_labels = [item['seg_label'] for item in batch]
    max_num_objects = max(label.shape[0] for label in labels)
    padded_labels = []
    for label in labels:
        pad_num = max_num_objects - label.shape[0]
        padded_label = F.pad(label, (0, 0, 0, pad_num), value=-1)
        padded_labels.append(padded_label)

    return torch.stack(imgs), torch.stack(padded_labels), torch.stack(seg_labels)

# def custom_collate(batch):
#     images = [item['image'] for item in batch]
#     detect_labels_batch_idx = [item['detect_label']['batch_idx'] for item in batch]
#     detect_labels_cls = [item['detect_label']['cls'] for item in batch]
#     detect_labels_bboxes = [item['detect_label']['bboxes'] for item in batch]
#     seg_labels = [item['seg_label'] for item in batch]

#     images = torch.stack(images, 0)
#     detect_labels_batch_idx = torch.cat(detect_labels_batch_idx, 0)
#     detect_labels_cls = torch.cat(detect_labels_cls, 0)
#     detect_labels_bboxes = torch.cat(detect_labels_bboxes, 0)
#     seg_labels = torch.stack(seg_labels, 0)

#     return {
#         'image': images,
#         'detect_label': {'batch_idx': detect_labels_batch_idx, 'cls': detect_labels_cls, 'bboxes': detect_labels_bboxes},
#         'seg_label': seg_labels
#     }

def custom_collate_noseg(batch):
    images = [item['image'] for item in batch]
    img_rel_path = [item['img_rel_path'] for item in batch]
    img_path = [item['img_path'] for item in batch]
    detect_labels_batch_idx = [item['detect_label']['batch_idx'] for item in batch]
    detect_labels_cls = [item['detect_label']['cls'] for item in batch]
    detect_labels_bboxes = [item['detect_label']['bboxes'] for item in batch]
    # seg_labels = [item['seg_label'] for item in batch]

    images = torch.stack(images, 0)
    detect_labels_batch_idx = torch.cat(detect_labels_batch_idx, 0)
    detect_labels_cls = torch.cat(detect_labels_cls, 0)
    detect_labels_bboxes = torch.cat(detect_labels_bboxes, 0)
    # seg_labels = torch.stack(seg_labels, 0)

    return {
        'image': images,
        'detect_label': {'batch_idx': detect_labels_batch_idx, 'cls': detect_labels_cls, 'bboxes': detect_labels_bboxes},
        'img_rel_path': img_rel_path,
        'img_path': img_path
        # 'seg_label': seg_labels
    }

def get_dataloader(cfgs, root, split='train', batch_size=4, num_workers=8, transform=None, shuffle=True):
    dataset = CustomDataset(cfgs, root, split, transform, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=   custom_collate, pin_memory=True)
    return dataloader

# def get_mot_dataloader(mot_root, yolo_annotation_root, split='train', batch_size=1, transform=None, shuffle=False):
#     dataset = Dataset_MOT(mot_root, yolo_annotation_root, split=split, transform=transform, batch_size=batch_size)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, collate_fn=custom_collate_noseg, pin_memory=True)
#     return dataset, dataloader


if __name__ == '__main__':
    # Example usage
    data_root = 'D:\Autonomous Research\DSPNet\data'

    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor()
    ])

    train_dataloader = get_dataloader(data_root, split='train', transform=transform)
    val_dataloader = get_dataloader(data_root, split='val', transform=transform)

    samples = next(iter(train_dataloader))
    image = transforms.ToPILImage()(samples['image'][0])
    image.show()
    print(samples['seg_label'].shape)
    print(samples['detect_label'])

