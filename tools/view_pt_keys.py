import torch

if __name__ == "__main__":
    dir = '/home/azuo/LRHPerception/outputs/all/checkpoints/ORI_distributed_step2199_loss6.289422113448381.pth'
    pretrained_dir = torch.load(dir, map_location='cpu')
    if 'model_state_dict' in pretrained_dir:
        pretrained_dir = pretrained_dir['model_state_dict']
    for k, v in pretrained_dir.items():
        print(k)