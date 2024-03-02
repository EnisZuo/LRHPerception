import torch
import torch.nn as nn
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from .swin_transformer import SwinTransformer
from .yolo_pafpn import YOLOPAFPN
from .yolo_head import YOLOXHead
from .vadepthnet import VADepthNet
from ..utils import postprocess
from ..utils.gru_cvae_utils import post_process as pred_post_process
from ..tracker.byte_tracker import BYTETracker
# from ..tracker.byte_tracker_ori import BYTETracker_ORI
from .GRU_CVAE import GRU_CVAE
from .seg_head import Segment

class DTPDNet(nn.Module):
    def __init__(self, cfgs, train_task='none'):
        # cfgs: cfgs
        super().__init__()
        self.cfgs = cfgs
        self.train_task = train_task
        self.track_results = []
        
        ini_cfgs = cfgs.MODEL
        pretrain_img_size = (352, 1216)
        patch_size = (4, 4)
        in_chans = 3
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
        window_size = 7

        # pretrain_img_size = img_size
        # patch_size = (4, 4)
        # in_chans = 3
        # embed_dim = 192
        # depths = [2, 2, 18, 2]
        # num_heads = [6, 12, 24, 48]
        # window_size = 12

        backbone_cfg = dict(
            pretrain_img_size=pretrain_img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            ape=True,
            drop_rate=0.
        )

        self.backbone = SwinTransformer(**backbone_cfg)
        self.backbone.init_weights(pretrained=cfgs.CKPT.SWIN_CKPT)
        
        self.neck = YOLOPAFPN(ini_cfgs.BACKBONE_DEPTH, ini_cfgs.WIDTH, ini_cfgs.BACKBONE_NECT_FEATURES, ini_cfgs.IN_CHANNELS, ini_cfgs.BACKBONE_DEPTHWISE)
        self.head = YOLOXHead(ini_cfgs.NUM_CLASSES, ini_cfgs.WIDTH, in_channels=ini_cfgs.IN_CHANNELS, cfgs=self.cfgs)
        self.seg_head = Segment(in_channel=int(ini_cfgs.IN_CHANNELS[0] * ini_cfgs.WIDTH), classes = 1)
        self.tracker = BYTETracker(ini_cfgs.TRACKER, self.cfgs.DATALOADER.RESIZE)
        self.traj_predictor = GRU_CVAE(ini_cfgs.GRU_CVAE)
        self.feature_extractor = nn.BatchNorm2d(ini_cfgs.IN_CHANNELS[0], affine=False)
        self.depth = VADepthNet(num_duo_conv=ini_cfgs.NUM_DUO_CONV_DPETH)
        
        if train_task == 'none':
            self.eval()
        elif train_task == 'det-seg':
            self.train()
            self.depth.eval()
        elif train_task == 'depth':
            print('depth set to train, other set to eval')
            self.eval()
            self.depth.train()
        
        self.yolo_process_time = 0
        self.tracker_process_time = 0
        self.traj_pred_process_time = 0
        self.depth_process_time = 0
        
        self.pri_img_parent_path = None
        self.frame_id = 0

    def forward(self, x, img_path=None, det_labels=None, depth_gt=None, specific = 'all'):
        trajectories, pred_goal, pred_traj, loss_dict = None, None, None, None
        backbone_out = self.backbone(x) # cfg.model.BACKBONE_NECT_FEATURES size: 1/(2^n)
        if specific == 'det':
            neck_out = self.neck([backbone_out[1], backbone_out[2], backbone_out[3]])
            return self.head(neck_out, det_labels, x)
        if specific == 'seg':
            neck_out = self.neck([backbone_out[1], backbone_out[2], backbone_out[3]])
            return self.seg_head(neck_out[0])
        if specific == 'depth':
            return self.depth([backbone_out[0], backbone_out[1], backbone_out[2], backbone_out[3]], depth_gt)
        neck_out = self.neck([backbone_out[1], backbone_out[2], backbone_out[3]])
        seg_head_out = self.seg_head(neck_out[0])
        yolox_head_out = self.head(neck_out, det_labels, x)
        depth_out = self.depth([backbone_out[0], backbone_out[1], backbone_out[2], backbone_out[3]], depth_gt)

        if yolox_head_out is not None:
            yolox_head_out = postprocess(yolox_head_out, self.cfgs.MODEL.NUM_CLASSES, self.cfgs.MODEL.CONFTHRESH, self.cfgs.MODEL.NMSTHRESH)
            dets = yolox_head_out[0]
        else:
            dets = None
        # print(dets.shape)
        if(dets is not None):
            tracker_out = self.run_tracker(self.cfgs, dets, img_path, x, feature_map=self.feature_extractor(backbone_out[1]), this_tracker=self.tracker)
            # print(len(tracker_out))
            # tracker_out_bbox = np.array([track.bboxes for track in tracker_out])
            # print(tracker_out_bbox)
            
            if(len(tracker_out) > 0):
                trajectories = self.tracker_out_post_process(tracker_out)
                # print(trajectories[:, -1])
                pred_goal, pred_traj, loss_dict = self.traj_predictor(trajectories.to(self.cfgs.DEVICE))
                # print(pred_goal, pred_traj)
                # print('pred_trag.shape: ', pred_traj)

                _, _, pred_goal, pred_traj = pred_post_process(self.cfgs, trajectories, trajectories, pred_traj, pred_goal)
                pred_traj = self.avg_pred_trag(pred_traj)

        return dets, seg_head_out.cpu(), trajectories, pred_goal, pred_traj, depth_out.cpu()
    
    def print_time_cost(self):
        print("time costs: ")
        print("yolo_precess_time: ", self.yolo_process_time)
        print("tracker_process_time: ", self.tracker_process_time)
        print("traj_pred_process_time: ", self.traj_pred_process_time)
        print("depth_process_time: ", self.depth_process_time)
   
    def avg_pred_trag(self, pred_traj):
        averaged_pred_trag = []
        for each_traget in pred_traj:
            each_timestep_avg = []
            for each_timestep in each_traget:
                b1 = np.mean(each_timestep[:, 0])
                b2 = np.mean(each_timestep[:, 1])
                b3 = np.mean(each_timestep[:, 2])
                b4 = np.mean(each_timestep[:, 3])
                each_timestep_avg.append([b1, b2, b3, b4])
            averaged_pred_trag.append(each_timestep_avg)
        return torch.tensor(averaged_pred_trag)
    
    def avg_pred_goal(self, pred_goal):
        averaged_pred_goal = []
        for each_traget in pred_goal:
            b1 = np.mean(each_traget[:, 0])
            b2 = np.mean(each_traget[:, 1])
            b3 = np.mean(each_traget[:, 2])
            b4 = np.mean(each_traget[:, 3])
            averaged_pred_goal.append([b1, b2, b3, b4])
        return np.array(averaged_pred_goal)
    
    def tracker_out_post_process(self, tracker_out):
        # extact all history trajecotry
        # tracker_out = sorted(tracker_out, key=lambda x: x.track_id)
        image_height = self.cfgs.DATALOADER.RESIZE[0]
        image_width = self.cfgs.DATALOADER.RESIZE[1]
        
        trajectories = []
        for track in tracker_out:
            bbox_history = track.bboxes.copy()
            for i, bbox in enumerate(bbox_history):
                if self.cfgs.DATASET.BBOX_TYPE == 'cxcywh':
                    tl_x, tl_y, w, h = bbox
                    cx = tl_x + w / 2.0
                    cy = tl_y + h / 2.0
                    if self.cfgs.DATASET.NORMALIZE:
                        cx /= image_width
                        cy /= image_height
                        w /= image_width
                        h /= image_height
                    bbox_history[i] = [cx, cy, w, h]
                else:
                    tl_x, tl_y, w, h = bbox
                    br_x = tl_x + w
                    br_y = tl_y + h
                    if self.cfgs.DATASET.NORMALIZE:
                        tl_x /= image_width
                        br_y /= image_height
                        br_x /= image_width
                        tl_y /= image_height
                    bbox_history[i] = [tl_x, tl_y, br_x, br_y]
            if (len(bbox_history) < self.cfgs.MODEL.GRU_CVAE.INPUT_LEN):
                padding = [bbox_history[0]] * (self.cfgs.MODEL.GRU_CVAE.INPUT_LEN - len(bbox_history))
                bbox_history = padding + bbox_history
            elif (len(bbox_history) > self.cfgs.MODEL.GRU_CVAE.INPUT_LEN):
                bbox_history = bbox_history[-self.cfgs.MODEL.GRU_CVAE.INPUT_LEN:]
            trajectories.append(bbox_history)
        return torch.tensor(trajectories, dtype=torch.float32)
  
    def run_tracker(self, cfgs, dets, img_path, img, feature_map, this_tracker):
        tracker_cfgs = cfgs.MODEL.TRACKER
        cur_img_parent_path = os.path.dirname(img_path)
        
        
        if self.pri_img_parent_path is None:
            self.pri_img_parent_path = cur_img_parent_path
            self.frame_id = 1
        elif self.pri_img_parent_path != cur_img_parent_path:
            results_filename = f"{self.pri_img_parent_path.split('/')[-2]}.txt"
            save_path = os.path.join(cfgs.TRAIN.OUTPUT_DIR, cfgs.TRAIN.TASKS, "tracking_results", results_filename)
            directory = os.path.dirname(save_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            self.write_results(save_path, self.track_results)
            
            self.track_results = []
            self.frame_id = 1
            self.pri_img_parent_path = cur_img_parent_path
        else:
            self.frame_id += 1
        
        if self.frame_id == 1:
            # this_tracker.print_sums()
            # print('previous video finished')
            self.tracker = BYTETracker(tracker_cfgs, img_size=self.cfgs.DATALOADER.RESIZE)
        
        tracks =  this_tracker.update(dets, img_size=self.cfgs.DATALOADER.RESIZE, img=img, feature_map=feature_map)
        
        tlwhs = []
        ids = []
        scores = []
        
        for each_track in tracks:
            tlwhs.append(each_track.tlwh)
            ids.append(each_track.track_id)
            scores.append(each_track.score)
            
        self.track_results.append((self.frame_id, tlwhs, ids, scores))
        
        return tracks
    
    def write_results(self, filename, results):
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids, scores in results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                    f.write(line)