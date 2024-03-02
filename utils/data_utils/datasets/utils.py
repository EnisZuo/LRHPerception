import numpy as np
import torch
import torch.nn.functional as F

class resize_img(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img_tensor):
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = F.interpolate(img_tensor, size=self.size, mode='bilinear', align_corners=False)
        img_tensor = img_tensor.squeeze(0)
        return img_tensor    
    
class gt_to_binary(object):
    def __call__(self, gt_tensor):
        gt_tensor = gt_tensor.permute(1, 2, 0)
        gt = gt_tensor.numpy()
        is_road = np.logical_and(gt[...,0] > gt[...,1], gt[...,0] > gt[...,2])
        is_road = ~ is_road
        road_mask = np.uint8(is_road)
        road_mask = torch.from_numpy(road_mask)
        road_mask = road_mask.unsqueeze(0)
        return road_mask