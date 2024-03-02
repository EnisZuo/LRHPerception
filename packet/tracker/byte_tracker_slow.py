import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import time

from .kalman_filter import KalmanFilter
from . import matching
from .basetrack import BaseTrack, TrackState
# from yolox.deepsort_tracker.deepsort import NearestNeighborDistanceMetric

# Past frame appearance 

# class Detection(object):
#     """
#     This class represents a bounding box detection in a single image.
#     Parameters
#     ----------
#     tlwh : array_like
#         Bounding box in format `(x, y, w, h)`.
#     confidence : float
#         Detector confidence score.
#     feature : array_like
#         A feature vector that describes the object contained in this image.
#     Attributes
#     ----------
#     tlwh : ndarray
#         Bounding box in format `(top left x, top left y, width, height)`.
#     confidence : ndarray
#         Detector confidence score.
#     feature : ndarray | NoneType
#         A feature vector that describes the object contained in this image.
#     """

#     def __init__(self, tlwh, confidence, feature):
#         self.tlwh = np.asarray(tlwh, dtype=np.float)
#         self.confidence = float(confidence)
#         self.feature = np.asarray(feature, dtype=np.float32)

#     def to_tlbr(self):
#         """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
#         `(top left, bottom right)`.
#         """
#         ret = self.tlwh.copy()
#         ret[2:] += ret[:2]
#         return ret

#     def to_xyah(self):
#         """Convert bounding box to format `(center x, center y, aspect ratio,
#         height)`, where the aspect ratio is `width / height`.
#         """
#         ret = self.tlwh.copy()
#         ret[:2] += ret[2:] / 2
#         ret[2] /= ret[3]
#         return ret

'''
TODO: 
combine Detection to STrack
update feature in STrack.update()
see the confidence scores, adjust max_update_feature distance
add feature cheack to the second association
'''

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, feature=None):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.bboxes = []
        
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.confidence = self.score
        self.tracklet_len = 0
        self.time_since_update = 0
        
        self.features = []
        if feature is not None:
            self.features.append(feature)
        # print('new strack{} initialized, features length', self.track_id, len(self.features))
        
        self.update_by_feature = False
        self.matched_det_idx = None
        

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))
        self.bboxes.append(self._tlwh)

        self.tracklet_len = 0
        self.time_since_update = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, update_by_feature=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        
        self.bboxes.append(new_track.tlwh)
        self.tracklet_len = 0
        self.time_since_update = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.confidence = self.score
        
        self.update_by_feature = update_by_feature
        if update_by_feature:
            self.features.append(new_track.feature[-1])

    def update(self, new_track, frame_id, update_by_feature=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.bboxes.append(new_track.tlwh)
        print(new_track.tlwh)
        
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.confidence = self.score
        
        # print('self.features', type(self.features))
        # print('new_track.features', type(new_track.features))
        # if (len(new_track.features) > 0):
        self.update_by_feature = update_by_feature
        if(update_by_feature):
            self.features.append(new_track.features[-1])

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BYTETracker(object):
    def __init__(self, cfgs):
        frame_rate = cfgs.FPS
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frist_match_sum = 0
        self.second_match_sum = 0
        self.frist_detection_sum = 0
        self.second_detection_sum = 0
        self.u_track_second = 0

        self.frame_id = 0
        self.cfgs = cfgs
        #self.det_thresh = args.track_thresh
        self.det_thresh = cfgs.TRACK_THRESH + 0.1
        self.buffer_size = int(frame_rate / 30.0 * cfgs.TRACK_BUFFER)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        
        self.first_match_thresh = 0.9
        self.second_match_thresh = 0.5
        self.iou_weight = 0.90
        self.max_update_feature_thresh = 0.21
        
        self.appearance_match = False
        self.update_feature = False
        
        print('tracker configs:')
        print('track_thresh: {}'.format(cfgs.TRACK_BUFFER))
        print('track_buffer: {}'.format(self.buffer_size))
        print('first_match_thresh: {}'.format(self.first_match_thresh))
        print('second_match_thresh: {}'.format(self.second_match_thresh))
        print('iou_weight: {}'.format(self.iou_weight))
        print('update_feature: {}'.format(self.update_feature))
        if (self.update_feature):
            print('max_update_feature_thresh: {}'.format(self.max_update_feature_thresh))
        
        self.deep_max_dist = 0.15
        self.deep_nn_budget = 100
        # self.deep_max_age = 70
        self.deep_match_sum = 0
        deep_max_cosine_distance = self.deep_max_dist
        self.metric = NearestNeighborDistanceMetric("cosine", deep_max_cosine_distance, self.deep_nn_budget)
        

    def update(self, output_results, img_info, img_size, frame=None, feature_map=None):
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if frame is None:
            # print(img_info)
            # print(img_size)
            # print(self.height, self.width)
            image_name = img_info[4][0]
            # print(image_name)
            img_root = './datasets/mot/train'
            img_path = os.path.join(img_root, image_name)
            current_image = cv2.imread(img_path)
            # print(current_image)
        else:
            current_image = frame
        self.height, self.width = img_info[0], img_info[1]
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        
        img_h, img_w = img_info[0], img_info[1]
        print(img_size[0], img_size[1])
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        print(bboxes)
        remain_inds = scores > self.cfgs.TRACK_THRESH
        inds_low = scores > 0.1
        inds_high = scores < self.cfgs.TRACK_THRESH

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []
        # print("detection_first num:", len(detections))
        # if len(detections) > 0:
        #     print('detection[:].bbox: ', detections[2]._tlwh)
        self.frist_detection_sum += len(detections)

        ''' Add newly detected tracklets to tracked_stracks'''
        # print('new iteration, self.tracked_stracks: ', len(self.tracked_stracks))
        
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        
        

        ''' Step 2: First association, with high score detection boxes''' 
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # print('len(strack_pool): ', len(strack_pool))
        
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        
        start_time = time.time()
        appearance_weight = 1 - self.iou_weight
        iou_dist = matching.iou_distance(strack_pool, detections)
        if len(strack_pool) > 0 and len(detections) > 0:
            strack_features = [track.features[-1] for track in strack_pool]
            det_features = self.get_feature_list_strack(detections, feature_map=feature_map)
            calc_start_time = time.time()
            temp_cost_matrix = self.metric.calc_cost_matrix_single_track_feature(strack_features, det_features)
            calc_end_time = time.time()
            temp_cost_matrix = self.normalize_cost_matrix(temp_cost_matrix)
            # print('iou_dist.shape, temp_cost_matrix.shape', iou_dist.shape, temp_cost_matrix.shape)
            # print(iou_dist)
            # print(temp_cost_matrix)
            dists = iou_dist * self.iou_weight + temp_cost_matrix * appearance_weight
            # end_time = time.time()
            # print('calc cost: ', (calc_end_time - calc_start_time) * 1000)
        else:
            dists = iou_dist
        end_time = time.time()
        # print('appearance cost cost: ', (end_time - start_time) * 1000)
        
        
        if not self.cfgs.MOT20:
            dists = matching.fuse_score(dists, detections)
        # calc feauture cost matric
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.cfgs.FIRST_MATCH_THRESH) # default 0.9
        # print('first matches: ', len(matches))
        # print('u_track_first: ', len(u_track))
        # print('u_detection_first: ', len(u_detection))
        self.frist_match_sum += len(matches)
        associated_activated_stracks = []
        associated_activated_det = []
        associated_refind_stracks = []
        associated_refind_det = []
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                # associated_activated_det.append(detections[idet])
                activated_stracks.append(track)
                track.matched_det_idx = idet
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                # associated_refind_det.append(det)
                refind_stracks.append(track)
                track.matched_det_idx = idet

        # temp_trust_strack, temp_untrust1 = self.compare_new_feature(associated_activated_stracks, current_img=current_image)
        # for i, track in enumerate(temp_trust_strack):
        #     track.update(detections[track.matched_det_idx], self.frame_id)
        #     activated_stracks.append(track)
        # temp_trust_strack, temp_untrust2 = self.compare_new_feature(associated_refind_stracks, current_img=current_image)
        # for i, track in enumerate(temp_trust_strack):
        #     track.re_activate(detections[track.matched_det_idx], self.frame_id, new_id=False)
        #     refind_stracks.append(track)
        # if len(temp_untrust1) > 0 or len(temp_untrust2) > 0:    
        #     print('temp_untrust1, temp_untrust2: ', temp_untrust1, temp_untrust2)
        
        
        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                          (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        # print('detection_second num:', len(detections_second))
        self.second_detection_sum += len(detections_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # r_tracked_stracks += temp_untrust1
        # r_tracked_stracks += temp_untrust2
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # print('r_tracked_stracks: ', len(r_tracked_stracks))
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=self.second_match_thresh)
        # print('u_track_second: ', u_track)
        self.u_track_second += len(u_track)
        # print('second matches: ', len(matches))
        self.second_match_sum += len(matches)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        
        associated_stracks = activated_stracks + refind_stracks
        
        
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                        
        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.cfgs.MOT20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        # print('initiate new stracks')
        # print(len(u_detection))
        initialized_stracks = []
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            initialized_stracks.append(track)
            # print('new track activated')
        
        self.update_features_strack(initialized_stracks, feature_map=feature_map)
            
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]   
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # # compare 1, 2, associations features, if the features of detections and tracks matches, update the stracks's features with new detection
        # # update feature for tracked targets in during 1, 2, associations
        
        if (self.update_feature):
            this_tracked_tlwh = [track._tlwh for track in associated_stracks]
            self.extract_feature(this_tracked_tlwh, feature_map=feature_map)
            # this_tracked_features = self._get_features(this_tracked_tlwh, current_image)
            this_tracked_features = self.extract_feature(this_tracked_tlwh, feature_map=feature_map)
            # print('associated_stracks: ', associated_stracks)
            trust_feature_tracks = initialized_stracks
            for i, track in enumerate(associated_stracks):
                dist = self.metric.temp_match(this_tracked_features[i], track.features[-1]) # TODO: update to every feature for a track
                if (dist < self.max_update_feature_thresh or track.score > self.det_thresh):
                    track.features.append(this_tracked_features[i])
                # else:
                #     # print(track.track_id, dist, 'feature is too far')

            # # update sample
            # if self.appearance_match:
            #     Keep_sample_tracks = self.tracked_stracks + self.lost_stracks
            #     keep_sample_id = [t.track_id for t in Keep_sample_tracks]
            #     # print('trust_feature_id: ', keep_sample_id)
            #     features, feature_id = [], []
            #     for track in trust_feature_tracks:
            #         # print('track.track_id: ', track.track_id)
            #         features += track.features
            #         feature_id += [track.track_id for _ in track.features]
            #         # print('track.features: ', track.features)
            #         # print(track.track_id, len(track.features))
            #         track.features = [track.features[-1]]
            #         # print(track.track_id, len(track.features))
            #     # print('targets: ', targets)
            #     self.metric.partial_fit(
            #         np.asarray(features), np.asarray(feature_id))
            #     self.metric.sample_filter(keep_sample_id)
        
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        # # print('self.tracked_stracks', self.tracked_stracks)
        # # print('self.lost_stracks', self.lost_stracks)
        
        return output_stracks
    
    def normalize_cost_matrix(sefl, cost_matrix):
        min_val = np.min(cost_matrix)
        max_val = np.max(cost_matrix)
        range_val = max_val - min_val
        normalized_matrix = (cost_matrix - min_val) / range_val
        return normalized_matrix
    
    def compare_new_feature(self, strack_pool, feature_map):
        trust_feature_tracks = []
        untrust_feature_tracks = []
        tlwhs = [track._tlwh for track in strack_pool]
        # features = self._get_features(tlwhs, current_img)
        features = self.extract_feature(tlwhs, feature_map=feature_map)
        for i, track in enumerate(strack_pool):
            dist = self.metric.temp_match(features[i], track.features[-1]) # TODO: update to every feature for a track
            if (dist < self.max_update_feature_thresh or track.score > self.det_thresh):
                # track.features.append(features[i])
                # print(track.track_id, dist, 'feature is good')
                trust_feature_tracks.append(track)
            else:
                # print(track.track_id, dist, 'feature is too far')
                untrust_feature_tracks.append(track)
        return trust_feature_tracks, untrust_feature_tracks
    
    def print_sums(self):
        print('detections: ', self.frist_detection_sum, self.second_detection_sum)
        print('matches: ', self.frist_match_sum, self.second_match_sum)
        print('u_track_second: ', self.u_track_second)
        print('deep_match_sum: ', self.deep_match_sum)
        
    # @staticmethod
    # def _xywh_to_tlwh(bbox_xywh):
    #     if isinstance(bbox_xywh, np.ndarray):
    #         bbox_tlwh = bbox_xywh.copy()
    #     elif isinstance(bbox_xywh, torch.Tensor):
    #         bbox_tlwh = bbox_xywh.clone()
    #     bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
    #     bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
    #     return bbox_tlwh


    # def _xywh_to_xyxy(self, bbox_xywh):
    #     x,y,w,h = bbox_xywh
    #     x1 = max(int(x-w/2),0)
    #     x2 = min(int(x+w/2),self.width-1)
    #     y1 = max(int(y-h/2),0)
    #     y2 = min(int(y+h/2),self.height-1)
    #     return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width - 1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height - 1)
        return x1,y1,x2,y2

    # def _xyxy_to_tlwh(self, bbox_xyxy):
    #     x1,y1,x2,y2 = bbox_xyxy

    #     t = x1
    #     l = y1
    #     w = int(x2-x1)
    #     h = int(y2-y1)
    #     return t,l,w,h
        
    
    def extract_feature(self, bbox_tlwh, feature_map):
        if not bbox_tlwh:  # Check if bbox_tlwh is empty
            return None  # or return torch.zeros(1, *feature_map.shape[1:], device=feature_map.device)
        
        avgpool = nn.AvgPool2d((8, 4), 1)
        crop_w = 4
        crop_h = 8
        half_w = int(crop_w / 2)
        half_h = int(crop_h / 2)
        padding = (half_w, half_w, half_h, half_h)
        
        start_time = time.time()
        map_height, map_width = feature_map.shape[-2], feature_map.shape[-1]
        feature_map = F.pad(feature_map, padding, mode='constant', value=0)
        
        crop_feature_list = []
        
        for box in bbox_tlwh:

            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            center_x = int(((x1 + x2) / 2) / self.width * map_width) + 2
            center_y = int(((y1 + y2) / 2) / self.height * map_height) + 4
            crop_x1 = int(center_x - crop_w / 2)
            crop_y1 = int(center_y - crop_h / 2)
            crop_x2 = int(center_x + crop_w / 2)
            crop_y2 = int(center_y + crop_h / 2)
        
            croped_feature = feature_map[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
            crop_feature_list.append(croped_feature)
            # print(croped_feature.shape)
        
        feature_crops = torch.cat(crop_feature_list, dim=0)
        # print(feature_crops.shape)
        out = avgpool(feature_crops).squeeze(-2, -1).cpu().numpy()
        # print(out.shape)
        end_time = time.time()
        # print('extract feature time cost: ', (end_time - start_time) * 1000)
        return out

    
    # def extract_feature(self, bbox_tlwh, feature_map):
    #     print(len(bbox_tlwh))
    #     if not bbox_tlwh:  # Check if bbox_tlwh is empty
    #         return None  # or return torch.zeros(1, *feature_map.shape[1:], device=feature_map.device)

    #     resize = (4, 8)
    #     avgpool = nn.AvgPool2d((8, 4), 1)
    #     # print(feature_map.shape)
    #     count_time = 0
        
    #     feature_crops = []
    #     map_height, map_width = feature_map.shape[-2], feature_map.shape[-1]
        
    #     for_start_time = time.time()
    #     for box in bbox_tlwh:
            
    #         x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
    #         # print(x2 - x1, y2 - y1)
    #         x1 = (int)((x1 / self.width) * map_width)
    #         y1 = (int)((y1 / self.height) * map_height)
    #         x2 = (int)((x2 / self.width) * map_width)
    #         y2 = (int)((y2 / self.height) * map_height)
            
    #         # crop from feature map
    #         # print(x1, y1, x2, y2)
    #         part_start_time = time.time()
    #         feature_crop = feature_map[:, :,y1:y2, x1:x2] # C, H, W
    #         part_end_time = time.time()
    #         feature_crop = feature_crop.squeeze(0)
    #         feature_crop = feature_crop.permute(1, 2, 0) # H, W, C
    #         # print(feature_crop.shape)
            
    #         # resize to fit average pooling
    #         feature_crop = feature_crop.cpu().numpy()
            
            
    #         feature_crop = cv2.resize(feature_crop.astype(np.float32)/255., resize)
            
    #         # print(feature_crop.shape)
    #         feature_crop = torch.from_numpy(feature_crop) # shape(8x4x320)
    #         feature_crop = feature_crop.permute(2, 0, 1)
    #         feature_crop = feature_crop.unsqueeze(0)
    #         # print(feature_crop.shape)
            
    #         # print(feature_crop.shape)
    #         feature_crops.append(feature_crop)
    #         count_time += (part_end_time - part_start_time) * 1000
    #         # print('part cost: ', (part_end_time - part_start_time) * 1000)
            
    #     print(count_time)
    #     for_end_time = time.time()
    #     # print('for loop cost: ', (for_end_time - for_start_time) * 1000)
            
    #     pool_start_time = time.time()
    #     feature_crops = torch.cat(feature_crops, dim=0)
    #     out = avgpool(feature_crops).squeeze(-2, -1).cpu().numpy()
    #     pool_end_time = time.time()
    #     # print('avgpool time cost: ', (pool_end_time - pool_start_time) * 1000)
    #     # print('extract feature cost: ', (end_time - start_time) * 1000)
    #     # print('out.shape: ', out[0].shape)
    #     return out
    
    def update_features_strack(self, strack_pool, feature_map):
        strack_tlwh = [track._tlwh for track in strack_pool]
        features_list = self.extract_feature(strack_tlwh, feature_map)
        for i, track in enumerate(strack_pool):
            # print(track.track_id, 'feature updated')
            track.features.append(features_list[i])
            
    def get_feature_list_strack(self, strack_pool, feature_map):
        track_tlwh = [track._tlwh for track in strack_pool]
        # print(len(track_tlwh))
        start_time = time.time()
        feature_list = self.extract_feature(track_tlwh, feature_map=feature_map)
        end_time = time.time()
        # print('get_feature_list cost: ', (end_time - start_time) * 1000)
        return feature_list
        

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
        
    def temp_match(self, detection_feature, track_feature):
            distance = self._metric([detection_feature], [track_feature])
            # print(distance)
            return distance

    def calc_cost_matrix_single_track_feature(self, track_features, det_features):
        cost_matrix = np.zeros((len(track_features), len(det_features)))
        for i, track_feature in enumerate(track_features):
            cost_matrix[i, :] = self._metric([track_feature], det_features)
        return cost_matrix
        
    def sample_filter(self, keep_sample_ids):
        self.samples = {k: self.samples[k] for k in keep_sample_ids}

    def partial_fit(self, features, targets):
        # print('targets in fit', targets)
        # print(len(features))
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            # print(target, 'added to sample.')
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]

    def distance(self, det_features, targets_id):
        cost_matrix = np.zeros((len(targets_id), len(det_features)))
        for i, target in enumerate(targets_id):
            # print('len(self.samples[target]), len(features)', len(self.samples[target]), len(features))
            cost_matrix[i, :] = self._metric(self.samples[target], det_features)
        return cost_matrix
    
    
def _cosine_distance(a, b, data_is_normalized=False):
    # print('(np.asarray(a)).shape, (np.asarray(b)).shape: ', (np.asarray(a)).shape, (np.asarray(b)).shape)
    
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    # print('a.shape, b.shape', a.shape, b.shape)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)
