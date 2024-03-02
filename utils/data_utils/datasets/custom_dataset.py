import os, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, split='train', transform=None, batch_size=16):
        self.batch_size = batch_size
        self.root = root
        self.split = split
        self.transform = transform
        self.index = 0

        self.image_folder = os.path.join(root, 'images', split)
        self.detect_labels_folder = os.path.join(root, 'detect_labels', split)
        self.seg_labels_folder = os.path.join(root, 'seg_labels', split)

        self.image_filenames = [f for f in os.listdir(self.detect_labels_folder) if f.endswith('.txt')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        self.index = 0 if self.index == self.batch_size else self.index
        img_name = self.image_filenames[idx].replace('.txt', '.png')
        img_path = os.path.join(self.image_folder, img_name)
        detect_label_path = os.path.join(self.detect_labels_folder, self.image_filenames[idx])
        seg_label_path = os.path.join(self.seg_labels_folder, self.image_filenames[idx])

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
            'seg_label': torch.tensor(seg_label, dtype=torch.float)
        }
        
        print(sample[image].shape)
        self.index += 1
        return sample
    

def custom_collate(batch):
    images = [item['image'] for item in batch]
    detect_labels_batch_idx = [item['detect_label']['batch_idx'] for item in batch]
    detect_labels_cls = [item['detect_label']['cls'] for item in batch]
    detect_labels_bboxes = [item['detect_label']['bboxes'] for item in batch]
    seg_labels = [item['seg_label'] for item in batch]

    images = torch.stack(images, 0)
    detect_labels_batch_idx = torch.cat(detect_labels_batch_idx, 0)
    detect_labels_cls = torch.cat(detect_labels_cls, 0)
    detect_labels_bboxes = torch.cat(detect_labels_bboxes, 0)
    seg_labels = torch.stack(seg_labels, 0)

    return {
        'image': images,
        'detect_label': {'batch_idx': detect_labels_batch_idx, 'cls': detect_labels_cls, 'bboxes': detect_labels_bboxes},
        'seg_label': seg_labels
    }

def get_dataloader(root, split='train', batch_size=4, num_workers=8, transform=None, shuffle=True):
    dataset = CustomDataset(root, split, transform, batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=custom_collate, pin_memory=True)
    return dataloader


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

