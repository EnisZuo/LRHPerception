import os
from PIL import Image
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from .yolox_utils.boxes import cxcywh_to_x1y1x2y2
import torch
# plot edges
edge_pairs = [(15,17), (15,0), (0,16), (16, 18), (0,1),
                (1,2), (1,5), (2,3), (3,4), (5,6), (6,7),
                (1,8), (8,9), (8,12), (9,10), (10,11),
                (12, 13), (13, 14), (11, 24,), (11, 22), (22, 23),
                (14, 21), (14, 19), (19, 20)]

def draw_single_pose(img, pose, color=None):
    '''
    Assume the poses are saved in BODY_25 format
    see here for details: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md#pose-output-format-body_25
    '''
    if color is None:
        color = np.random.rand(3) * 255
    
    # if isinstance(pose, torch.Tensor):
    #     pose = pose.type(torch.int)
    if isinstance(pose, np.ndarray):
        pose = pose.astype(np.int)
    else:
        raise TypeError('Unknown pose type {}'.format(type(pose)))    
    # plot points
    for point in pose:
        if point.max() > 0:
            cv2.circle(img, tuple(point.tolist()), radius=3, color=color, thickness=-1)
    
    for edge in edge_pairs:
        if pose[edge[0]].max() <= 0 or pose[edge[1]].max() <= 0:
            continue
        else:
            cv2.line(img, tuple(pose[edge[0]].tolist()), tuple(pose[edge[1]].tolist()), color=color, thickness=2)
    return img

def vis_pose_on_img(img, poses, color=None):
    '''skeleton_traj: (T, 50)'''
    # visualize
    for pose in poses: #inversed_X_merged: #: #: 
        pose = pose.reshape(-1, 2)
        img = draw_single_pose(img, pose, color)#, dotted=False)

    return img

def viz_pose_trajectories(poses, img_root, vid_traj_id, frame_id, img=None, color=None):
    '''
    draw the temporal senquence of poses
    poses: (T, 25, 2)
    img: np.array
    '''
    frame_id = int(frame_id[-1])
    # NOTE this only works for JAAD
    vid = vid_traj_id[:10]
    traj_id = vid_traj_id[11:]
    frames_path = os.path.join(img_root, vid)
    if img is None:
        img = Image.open(os.path.join(frames_path, str(frame_id).zfill(5)+'.png'))
        img = np.array(img)
    
    img = vis_pose_on_img(img, poses, color=color)
    
    return img


class Visualizer():
    def __init__(self, cfgs, mode='image', save_frame_dir=None):
        self.cfgs = cfgs
        self.mode = mode
        self.save_frame_dir = save_frame_dir
        self.frame_count = 0
        if self.mode == 'image':
            self.img = None
        elif self.mode == 'plot':
            self.fig, self.ax = None, None
        else:
            raise NameError(mode)
            
    def initialize(self, img_path=None):
        if self.mode == 'image':
            self.img = np.array(Image.open(img_path))
            self.img = cv2.resize(self.img, (self.cfgs.DATALOADER.RESIZE[1], self.cfgs.DATALOADER.RESIZE[0]), interpolation = cv2.INTER_LINEAR)
            self.H, self.W, self.CH = self.img.shape
        elif self.mode == 'plot':
            self.fig, self.ax = plt.subplots()
    
    def visualize(self, 
                  inputs, 
                  id_to_show=0,
                  normalized=False, 
                  bbox_type='x1y1x2y2',
                  color=(255,0,0), 
                  thickness=4, 
                  radius=5,
                  label=None,  
                  viz_type='point', 
                  viz_time_step=None):
        if viz_type == 'bbox':
            self.viz_bbox_trajectories(inputs, normalized=normalized, bbox_type=bbox_type, color=color, viz_time_step=viz_time_step)
        elif viz_type == 'point':
            self.viz_point_trajectories(inputs, color=color, label=label, thickness=thickness, radius=radius)
    
    def clear(self):
        plt.close()
        # plt.cla()
        # plt.clf()
        self.fig.clear()
        self.ax.clear()
        del self.fig, self.ax
    
    def save_plot(self, fig_path, clear=True):
        self.ax.set_xlabel('x [m]', fontsize=12)
        self.ax.set_ylabel('y [m]', fontsize=12)
        self.ax.legend(fontsize=12)
        plt.savefig(fig_path)
        if clear:
            self.clear()

    def plot_to_image(self, clear=False):
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        self.ax.legend()
        # draw the renderer
        self.fig.canvas.draw()
        # Get the RGBA buffer from the figure
        w,h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf.shape = (h, w, 3)
        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = np.roll ( buf, 3, axis = 2 )
        if clear:
            self.clear()
        return buf

        # pdb.set_trace()
    def viz_point_trajectories(self, points, color=(255,0,0), label=None, thickness=4, radius=5):
        '''
        points: (T, 2) or (T, K, 2)
        '''
        if self.mode == 'image':
            # plot traj on image
            if len(points.shape) == 2:
                points = points[:, None, :]
            T, K, _ = points.shape
            points = points.astype(np.int32)
            for k in range(K):
                # pdb.set_trace()
                cv2.polylines(self.img, [points[:, k, :]], isClosed=False, color=color, thickness=thickness)
                    
                for t in range(T):
                    cv2.circle(self.img, tuple(points[t, k, :]), color=color, radius=radius, thickness=-1)
        elif self.mode == 'plot':
            # plot traj in matplotlib 
            # pdb.set_trace()
            if len(points.shape) == 2:
                self.ax.plot(points[:, 0], points[:, 1], '-o', color=color, label=label)
            elif len(points.shape) == 3:
                # multiple traj as (T, K, 2)
                for k in range(points.shape[1]):
                    label = label if k == 0 else None
                    self.ax.plot(points[:, k, 0], points[:, k, 1], '-', color=color, label=label)
            else:
                raise ValueError('points shape wrong:', points.shape)
            self.ax.axis('equal')

    def draw_single_bbox(self, bbox, color=None):
        '''
        img: a numpy array
        bbox: a list or 1d array or tensor with size 4, in x1y1x2y2 format
        
        '''
        
        if color is None:
            color = np.random.rand(3) * 255
        if color == (255., 0, 0):
            cv2.rectangle(self.img, (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), color, 2)
        else:
            cv2.rectangle(self.img, (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), color, 1)
    
    def viz_bbox_trajectories(self, bboxes, normalized=False, bbox_type='x1y1x2y2', color=None, thickness=2, radius=2, viz_time_step=None):
        '''
        bboxes: (T,4) or (T, K, 4)
        '''
        if len(bboxes.shape) == 2:
            bboxes = bboxes[:, None, :]

        if normalized:
            bboxes[...,[0, 2]] *= self.W
            bboxes[...,[1, 3]] *= self.H
        if bbox_type == 'cxcywh':
            bboxes = cxcywh_to_x1y1x2y2(bboxes)
        elif bbox_type == 'x1y1x2y2':
            pass
        else:
            raise ValueError(bbox_type)
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.numpy()
        bboxes = bboxes.astype(np.int32)
        T, K, _ = bboxes.shape

        # also draw the center points
        center_points = (bboxes[..., [0, 1]] + bboxes[..., [2, 3]])/2 # (T, K, 2)
        self.viz_point_trajectories(center_points, color=color, thickness=thickness, radius=radius)

        # draw way point every several frames, just to make it more visible
        if viz_time_step:
            bboxes = bboxes[viz_time_step, :]
            T = bboxes.shape[0]
        for t in range(T):
            for k in range(K):
                self.draw_single_bbox(bboxes[t, k, :], color=color)
                if T == 2:
                    cv2.putText(self.img, f'+{(t+1)*15/10}S', (bboxes[t, k, 0], bboxes[t, k, 1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
    def viz_goal_map(img, goal_map):
        '''
        img:
        goal_map: goal map after sigmoid
        '''
        alpha = 0.5
        # NOTE: de-normalize the image 
        img = copy.deepcopy(img)
        img = img.transpose((1,2,0))
        img = (((img + 1)/2) * 255).astype(np.uint8)
        
        cm = plt.get_cmap('coolwarm')
        goal_map = (cm(goal_map)[:, :, :3] * 255).astype(np.uint8)
        # img = cv2.addWeighted(goal_map, alpha, img, 1 - alpha, 0, img)
        
        return goal_map

    def viz_depth(self, depth_tensor):
        depth_tensor = depth_tensor.squeeze(0, 1)
        # Normalize the depth values to 0-1 range
        depth_numpy = depth_tensor.cpu().detach().numpy()
        depth_numpy = depth_numpy
        
        depth_tensor = (depth_tensor - torch.min(depth_tensor)) / (torch.max(depth_tensor) - torch.min(depth_tensor))
        # Convert the depth tensor to numpy for visualization and ensure it's detached from any computations
        # Convert to a format that cv2.applyColorMap requires
        depth_numpy_uint8 = cv2.normalize(depth_numpy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # Apply colormap
        depth_colored = cv2.applyColorMap(-depth_numpy_uint8, cv2.COLORMAP_PLASMA)
        alpha = 0.25
        beta = 1.0 - alpha
        self.img = cv2.addWeighted(self.img, alpha, depth_colored, beta, 0.0)
        return depth_colored
    
    def viz_seg(self, seg_mask):
        seg_mask = seg_mask.squeeze()
        palette = np.array(
            [[0, 0, 0],
            [255, 255, 255]]
        )
        seg_mask[seg_mask > 0.5] = 1
        seg_mask[seg_mask <= 0.5] = 0
        color_seg = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
        for i, color in enumerate(palette):
            color_seg[seg_mask == i, :] = color

        color_seg = color_seg[..., ::-1]  # convert to BGR
        # print(one_seg)
        color_mask = np.mean(color_seg, 2)
        self.img[color_mask != 0] = self.img[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5

        # self.img = color_seg.astype(np.uint8)
        
    def viz_det(self, bboxes):
        # Define a color for each class BGR
        colors = [(200, 200, 0),   # light blue 
          (0, 200, 200),   # yellow
          (200, 0, 200),   # pink
          (200, 100, 50),  # peach
          (50, 200, 100)]  # sea green
        
        class_to_idx = {'Car': 0, 'Van': 1, 'Truck': 2, 'Pedestrian': 3, 'Cyclist': 4}
        idx_to_class = {class_to_idx[k]: k for k in class_to_idx}

        # Convert the image to OpenCV format (if it's not already)

        # Draw each bbox

        for bbox in bboxes:
            tlx, tly, brx, bry, conf, _, cls_idx = bbox
            if conf < 0.8:
                continue
            # Convert coordinates to integers
            tlx, tly, brx, bry = map(int, [tlx, tly, brx, bry])
            # print(tlx, tly, brx, bry, conf, cls_idx)
            # Choose color based on class
            color = colors[int(cls_idx)%len(colors)]
            # Draw bbox
            cv2.rectangle(self.img, (tlx, tly), (brx, bry), color, 1)
            # Add class label
            cv2.putText(self.img, str(idx_to_class[cls_idx]), (tlx, tly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

def viz_results(viz,
                dets=None,
                seg_mask=None,
                X_global=None, #his trajectory
                y_global=None, 
                pred_traj=None, 
                depth_map=None,
                img_path=None,
                bbox_type='cxcywh',
                normalized=True,
                logger=None, 
                name=''):
    '''
    given prediction output, visualize them on images or in matplotlib figures.
    '''
    # 1.1 initialize visualizer
    viz.initialize(img_path)

    # depth_map = viz.viz_depth(depth_map)  
    if dets is not None:
        viz.viz_det(dets.cpu().numpy())
    
    if pred_traj is not None: 
        pred_traj = pred_traj[:, :30]
        for id_to_show in range(pred_traj.shape[0]):
            # print(X_global[id_to_show][0], X_global[id_to_show][-1])
            viz.visualize(pred_traj[id_to_show], color=(0., 0., 255.), label='pred future', viz_type='bbox', 
                        normalized=normalized, bbox_type=bbox_type, viz_time_step=[14, 29])
    if X_global is not None:
        for id_to_show in range(X_global.shape[0]):
            viz.visualize(X_global[id_to_show], color=(255., 0., 0.), label='past', viz_type='bbox', 
                        normalized=True, bbox_type=bbox_type, viz_time_step=[0])
    if seg_mask is not None:
        viz.viz_seg(seg_mask)

    depth_map = viz.viz_depth(depth_map)  
    viz_img = viz.img
    
    # if depth_map.ndim != 3:
    #     print(depth_map.shape)
    cv2.imwrite(f'{viz.save_frame_dir}/{viz.frame_count:05d}.png', viz_img)
    # print(f'image write to {viz.save_frame_dir}/{viz.frame_count:05d}.png')
    viz.frame_count += 1

    if hasattr(logger, 'log_image'):
        logger.log_image(viz_img, label=name)
        # logger.log_image(depth_map, label='depth map')