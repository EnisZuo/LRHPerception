import torch
import torch.nn as nn
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from .darknet import CSPDarknet
from .yolo_pafpn import YOLOPAFPN
from .yolo_head import YOLOXHead
from .vadepthnet import VADepthNet
from ..utils import postprocess
from ..utils.gru_cvae_utils import post_process as pred_post_process
from ..tracker.byte_tracker import BYTETracker
from collections import defaultdict
from .GRU_CVAE import GRU_CVAE
from .seg_head import Segment

class DTPDNet(nn.Module):
    def __init__(self, cfgs, train_task='none'):
        super().__init__()
        self.cfgs = cfgs
        ini_cfgs = cfgs.MODEL
        self.train_task = train_task
        self.backbone = CSPDarknet(ini_cfgs.BACKBONE_DEPTH, ini_cfgs.WIDTH, depthwise=ini_cfgs.BACKBONE_DEPTHWISE, act=ini_cfgs.BACKBONE_ACT, out_features=ini_cfgs.BACKBONE_NECT_FEATURES)
        self.neck = YOLOPAFPN(ini_cfgs.BACKBONE_DEPTH, ini_cfgs.WIDTH, ini_cfgs.BACKBONE_NECT_FEATURES, ini_cfgs.IN_CHANNELS, ini_cfgs.BACKBONE_DEPTHWISE)
        self.head = YOLOXHead(ini_cfgs.NUM_CLASSES, ini_cfgs.WIDTH, in_channels=ini_cfgs.IN_CHANNELS)
        self.seg_head = Segment(in_channel=int(ini_cfgs.IN_CHANNELS[0] * ini_cfgs.WIDTH), classes = 1)
        self.tracker = BYTETracker(ini_cfgs.TRACKER)
        self.traj_predictor = GRU_CVAE(ini_cfgs.GRU_CVAE)
        self.depth = VADepthNet()
        
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

    def forward(self, x, img_path=None, det_labels=None, depth_gt=None):
        trajectories, pred_goal, pred_traj, loss_dict = None, None, None, None
        # yolo_start_time = time.time()
        backbone_out = self.backbone(x) # cfg.model.BACKBONE_NECT_FEATURES size: 1/(2^n)
        neck_out = self.neck([backbone_out['dark3'], backbone_out['dark4'], backbone_out['dark5']])
        depth_out = self.depth([backbone_out['dark2'], backbone_out['dark3'], backbone_out['dark4'], backbone_out['dark5']], depth_gt)
        if (self.train_task == 'depth'):
            return depth_out
        yolox_head_out = self.head(neck_out, det_labels)
        seg_head_out = self.seg_head(neck_out[0])
        if (self.train_task == 'det-seg'):
            return yolox_head_out

        # self.yolo_process_time += time.time() - yolo_start_time
        yolox_head_out = postprocess(yolox_head_out, self.cfgs.MODEL.NUM_CLASSES, self.cfgs.MODEL.CONFTHRESH, self.cfgs.MODEL.NMSTHRESH)
        dets = yolox_head_out[0]
        if(dets is not None):
            # tracker_start_time = time.time()
            tracker_out = self.run_tracker(self.cfgs, dets, img_path, feature_map=self.backbone.extract_feature_map(), this_tracker=self.tracker)
            # self.tracker_process_time += time.time() - tracker_start_time
            # print(len(tracker_out))
            # tracker_out_bbox = np.array([track.bboxes for track in tracker_out])
            # print(tracker_out_bbox)
        
            if(len(tracker_out) > 0):
                trajectories = self.tracker_out_post_process(tracker_out)
                # print('trajectories.shape: ', trajectories.shape)
                # traj_pred_start_time = time.time()
                pred_goal, pred_traj, loss_dict = self.traj_predictor(trajectories.to(self.cfgs.DEVICE))
                # self.traj_pred_process_time += time.time() - traj_pred_start_time
                # print('pred_trag.shape: ', pred_traj)
                _, _, pred_goal, pred_traj = pred_post_process(self.cfgs, trajectories, trajectories, pred_traj, pred_goal)
                # print('pred_trag.shape: ', pred_traj)
                if not self.training and len(pred_traj) > 0:
                    pred_goal = self.avg_pred_goal(pred_goal)
                    pred_traj = self.avg_pred_trag(pred_traj)
                    # print('averaged_pred_goal.shape: ', pred_goal.shape)
                    # print('averaged_pred_trag.shape: ', pred_traj)
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
        return np.array(averaged_pred_trag)
    
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
                    if self.cfgs.DATASET.NORMALIZE == 'zero-one':
                        cx /= image_width
                        cy /= image_height
                        w /= image_width
                        h /= image_height
                    bbox_history[i] = [cx, cy, w, h]
                else:
                    tl_x, tl_y, w, h = bbox
                    br_x = tl_x + w
                    br_y = tl_y + h
                    if self.cfgs.DATASET.NORMALIZE == 'zero-one':
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
  
    def run_tracker(self, cfgs, dets, img_path, feature_map, this_tracker):
        tracker_cfgs = cfgs.MODEL.TRACKER
        cur_img_parent_path = os.path.dirname(img_path)
        
        if self.pri_img_parent_path is None:
            self.pri_img_parent_path = cur_img_parent_path
        elif self.pri_img_parent_path != cur_img_parent_path:
            self.frame_id = 1
            self.pri_img_parent_path = cur_img_parent_path
        else:
            self.frame_id += 1
        
        if self.frame_id == 1:
            # this_tracker.print_sums()
            # print('previous video finished')
            self.tracker = BYTETracker(tracker_cfgs)
        
        return this_tracker.update(dets, img_size=self.cfgs.DATALOADER.RESIZE , feature_map=feature_map)
    
    
        # if video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
        #     tracker_cfgs.TRACK_BUFFER = 14
        # elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
        #     tracker_cfgs.TRACK_BUFFER = 25
        # else:
        #     tracker_cfgs.TRACK_BUFFER = 30

        # if video_name == 'MOT17-01-FRCNN':
        #     tracker_cfgs.TRACK_THRESH = 0.65
        # elif video_name == 'MOT17-06-FRCNN':
        #     tracker_cfgs.TRACK_THRESH = 0.65
        # elif video_name == 'MOT17-12-FRCNN':
        #     tracker_cfgs.TRACK_THRESH = 0.7
        # elif video_name == 'MOT17-14-FRCNN':
        #     tracker_cfgs.TRACK_THRESH = 0.67
        # elif video_name in ['MOT20-06', 'MOT20-08']:
        #     tracker_cfgs.TRACK_THRESH = 0.3
        # else:
        #     tracker_cfgs.TRACK_THRESH = ori_thresh
        # # if video_name not in video_names:
        # #     video_names[video_id] = video_name