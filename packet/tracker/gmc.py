import os.path
import cv2
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import torch.nn.functional as F
class GMC:
    def __init__(self, args, method='sparseOptFlow', downscale=2, verbose=None, video_name=None, img_size=None):
        super(GMC, self).__init__()
        self.video_name = video_name
        self.img_size = img_size
        self.method = method
        self.args = args
        self.range = 7
        self.conv_thresh = 0.9
        self.top_points = 210
        self.downscale = max(1, int(downscale))


        if self.method == 'sparseOptFlow' or self.method == 'featureMap':
            self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3,
                                       useHarrisDetector=False, k=0.04)
            # self.gmc_file = open('GMC_results.txt', 'w')

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

        self.iso_point_kernal = torch.tensor([[1, 1, 1],
                        [1, -8, 1],
                        [1, 1, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda:0")
        
        self.hori_line_kernal = torch.tensor([[-1, -1, -1,],
                                   [2, 2, 2],
                                   [-1, -1, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to("cuda:0")

        mask_size = (img_size[0] // self.downscale, img_size[1] // self.downscale)
        self.mask = torch.zeros(mask_size, dtype=torch.uint8).to("cuda:0")

        # Calculate the height of the upper half
        upper_half_height = mask_size[0] // 2
        
        # Set the mask to 0 for the upper half of the frame
        self.mask[:upper_half_height, :] = 1
            
        self.counter = 1

    def apply(self, raw_frame, bbox=None):
        if self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, bbox)
        elif self.method == 'none':
            return np.eye(2, 3)
        elif self.method == "featureMap":
            return self.applyFeautreMap(raw_frame, bbox)
        else:
            return np.eye(2, 3)

    def applySparseOptFlow(self, raw_frame, bboxes=None):

        t0 = time.time()

        # Initialize
        height, width, _ = raw_frame.shape
        #frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        frame = torch.mean(raw_frame.float(), dim=2)
        H = np.eye(2, 3)

        # Downscale image
        # if self.downscale > 1.0:
        #     # frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
        #     #frame = t
        frame = frame.unsqueeze(0).unsqueeze(0)
        frame = F.interpolate(frame, size=(height // self.downscale, width // self.downscale), mode='bicubic')
        frame = frame.squeeze(0).squeeze(0)

        # find the keypoints
        # start_time = time.time()
        keypoints = self.conv_points(frame, self.iso_point_kernal, self.mask)
    
        
        frame = frame.detach().cpu().numpy().astype(np.uint8)   
        # Handle first frame
        if not self.initializedFirstFrame:
            # Initialize data
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            # Initialization done
            self.initializedFirstFrame = True

            return H

        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(self.prevFrame, frame, self.prevKeyPoints, None)
        
        # visualize keypoints -------------------------------
        # # Convert image to color if it's grayscale to see the keypoints clearly


        # leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(prevPoints, 0)):
            H, inliesrs = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            # Handle downscale
            if self.downscale > 1.0:
                H[0, 2] *= self.downscale
                H[1, 2] *= self.downscale
        else:
            print('Warning: not enough matching points')

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        t1 = time.time()

        # gmc_line = str(1000 * (t1 - t0)) + "\t" + str(H[0, 0]) + "\t" + str(H[0, 1]) + "\t" + str(
        #     H[0, 2]) + "\t" + str(H[1, 0]) + "\t" + str(H[1, 1]) + "\t" + str(H[1, 2]) + "\n"
        # self.gmc_file.write(gmc_line)

        return H


    def makeDetMask(self, frame, bboxes):
        mask = torch.ones(frame.shape[:2], dtype=torch.uint8)
        for x1, y1, x2, y2 in torch.tensor(bboxes, dtype=torch.uint8):
            mask[y1:y2, x1:x2] = 0

        return mask
    
    def makeFrameMask(self, frame):
        mask = torch.zeros(frame.shape[:2], dtype=torch.uint8)
        # for x1, y1, x2, y2 in np.array(bboxes).astype(int):
        #     mask[y1:y2, x1:x2] = 0
        
        # Calculate the height of the upper half
        upper_half_height = frame.shape[0] // 2
        left_split = int(frame.shape[1] * 3 / 10)
        right_split = int(frame.shape[1] * 7 / 10)
        
        # Set the mask to 0 for the upper half of the frame
        mask[:upper_half_height, :] = 1
        
        # mask[:upper_half_height, :left_split] = 1
        # mask[:upper_half_height, right_split:] = 1

        return mask

    
    def conv_points(self, frame, kernal, mask=None):
        frame = torch.tensor(frame, dtype=torch.float32)
        frame = frame.unsqueeze(0).unsqueeze(0)
        frame = frame.to("cuda:0")
        
        result = F.conv2d(frame, kernal, padding=1)

        threshold_value = self.conv_thresh  # Adjust threshold as necessary
        result = (result > threshold_value).int()
        
        # If a mask is provided, apply it
        if mask is not None:
            result = result & mask
        
        # Get the indices of non-zero elements
        y_indices, x_indices = (result[0][0] > 0).nonzero(as_tuple=True)
        points = torch.stack((x_indices, y_indices), dim=1)    
        
        if not self.range > 0:
            if len(points) > self.top_points:
                points = points[:self.top_points]
            return points.cpu().numpy().reshape(-1, 1, 2).astype(np.float32)
        
        else:
            points = points[::self.range]
            if len(points) > self.top_points:
                points = points[:self.top_points]
                points = points.cpu().numpy().reshape(-1, 1, 2).astype(np.float32)

            return points
    