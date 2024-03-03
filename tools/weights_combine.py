import torch
import os
if __name__ == "__main__":
    original_ckpt_path = '/home/azuo/LRHPerception/checkpoints/ORI.pth'
    new_ckpt_path = '/home/azuo/LRHPerception/checkpoints/seg_loss_0.3603168725967407_mIOU_0.8931286334991455.pth'
    output_ckpt_path = '/home/azuo/LRHPerception/checkpoints/ORI_updated_seg.pth'
    
    original_ckpt = torch.load(original_ckpt_path, map_location='cpu')
    new_seg_head_ckpt = torch.load(new_ckpt_path, map_location='cpu')
    
    if 'model_state_dict' in original_ckpt:
        original_ckpt = original_ckpt['model_state_dict']
    if 'model_state_dict' in new_seg_head_ckpt:
        new_seg_head_ckpt = new_seg_head_ckpt['model_state_dict']
    
    for k, v in new_seg_head_ckpt.items():
        if k.startswith('seg_head'):  # Ensure we're only updating seg_head weights
            ori_key = f'module.{k}'
            original_ckpt[ori_key] = v
            
    torch.save(original_ckpt, output_ckpt_path)
    print(f"Updated checkpoint saved to {output_ckpt_path}")
    
    