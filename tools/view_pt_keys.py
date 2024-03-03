import torch

if __name__ == "__main__":
    dir = '/home/azuo/LRHPerception/checkpoints/seg_loss_0.3603168725967407_mIOU_0.8931286334991455.pth'
    pretrained_dir = torch.load(dir, map_location='cpu')
    if 'model_state_dict' in pretrained_dir:
        pretrained_dir = pretrained_dir['model_state_dict']
    for k, v in pretrained_dir.items():
        print(k)