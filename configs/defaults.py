import os
from yacs.config import CfgNode as CN

_C = CN()

_C.NUM_GPUS = 1
_C.DEVICE = 'cuda:1'
_C.GPU = 1
_C.RANK = 0
_C.NUM_THREADS = 1
_C.DIST_BACKEND = 'nccl'
_C.DIST_URL = 'tcp://127.0.0.1:1234'
_C.LOG_FREQ = 200
_C.NGPU_PER_NODE = 3
_C.DISTRIBUTED = False
_C.MULTIPROCESSING_DISTRIBUTED = False
_C.MODEL_NAME = 'LRHPerception'
_C.PREDICTION_METHOD = 'GRU_CVAE'

_C.USE_WANDB = False
_C.VISUALIZE = True
_C.PROJECT = 'LRHPerception'
_C.PRINT_TIME = False

# -------CKPT-------
_C.CKPT = CN()
_C.CKPT.LOG_DIR = '/home/azuo/LRHPerception/outputs/logs'
_C.CKPT.SWIN_CKPT = '/home/azuo/Downloads/swin_tiny_patch4_window7_224_22k.pth'
_C.CKPT.ORI_CKPT = '/home/azuo/LRHPerception/checkpoints/ORI_updated_seg.pth'
# _C.CKPT.DEPTH_CKPT = '/home/azuo/LRHPerception/outputs/all/checkpoints/ORI_distributed_step2199_loss6.289422113448381.pth'
# _C.CKPT.DETECT_CKPT = '/home/azuo/LRHPerception/packet/checkpoint/bytetrack_x_mot17.pth-'
_C.CKPT.TRAJ_PRED_CKPT = '/home/azuo/LRHPerception/packet/checkpoint/PIE_Epoch_038.pth'
# _C.CKPT.SEG_HEAD_CKPT = '/home/azuo/LRHPerception/outputs/checkpoints/kitti_road/2023-06-07-17-24-56/epoch_49.pth'

# -------DATASET-------
_C.DATASET = CN()
_C.DATASET.ROOT = '/home/azuo/LRHPerception/datasets'

_C.DATASET.DEPTH = CN()
_C.DATASET.DEPTH.FILENAMES_FILE_EVAL = '/home/azuo/Trajectory_Prediction/VA-DepthNet/data_splits/eigen_test_files_with_gt.txt'
_C.DATASET.DEPTH.FILENAMES_FILE = '/home/azuo/Trajectory_Prediction/VA-DepthNet/data_splits/eigen_train_files_with_gt.txt'
_C.DATASET.DEPTH.USE_RIGHT = False
_C.DATASET.DEPTH.DO_RANDOM_ROTATE = False
_C.DATASET.DEPTH.DEGREE = 0
_C.DATASET.DEPTH.RESIZE = (352, 1216)
_C.DATASET.DEPTH.DATA_PATH = '/home/azuo/Trajectory_Prediction/Monocular-Depth-Estimation-Toolbox/data/kitti/input'
_C.DATASET.DEPTH.DATA_PATH_EVAL = '/home/azuo/Trajectory_Prediction/Monocular-Depth-Estimation-Toolbox/data/kitti/input'
_C.DATASET.DEPTH.GT_PATH = '/home/azuo/Trajectory_Prediction/Monocular-Depth-Estimation-Toolbox/data/kitti/gt_depth'
_C.DATASET.DEPTH.GT_PATH_EVAL = '/home/azuo/Trajectory_Prediction/Monocular-Depth-Estimation-Toolbox/data/kitti/gt_depth'

_C.DATASET.DETECT = CN()
_C.DATASET.DETECT.DET_ROOT = '/home/azuo/LRHPerception/datasets/kitti/detection_2d'
_C.DATASET.DETECT.NAME = 'city'

_C.DATASET.KITTI_RAW = CN()
_C.DATASET.KITTI_RAW.ROOT = '/home/azuo/Trajectory_Prediction/Monocular-Depth-Estimation-Toolbox/data/kitti'

_C.DATASET.KITTI_SEG = CN()
_C.DATASET.KITTI_SEG.ROOT = '/home/azuo/LRHPerception/datasets/kitti/data_semantics'

_C.DATASET.CITY_ROOT = '/home/azuo/DSPNet/data'

# -------DATASET cfg for predictor-------
_C.DATASET.MIN_BBOX = [0, 0, 0, 0]
_C.DATASET.MAX_BBOX = [1216, 352, 1216, 352] # [512, 512, 512, 512]
_C.DATASET.BBOX_TYPE = 'x1y1x2y2' # bbox is in cxcywh format
_C.DATASET.NORMALIZE = True # normalize to 0-1 'zero-one'
_C.DATASET.KITTI_ROOT = '/home/azuo/LRHPerception/datasets/kitti'

# ---- DATALOADER -----
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 1
_C.DATALOADER.RESIZE = (352, 1216)

# -------MODEL-------
_C.MODEL = CN()
_C.MODEL.DEVICE = 'cuda:1'
_C.MODEL.WIDTH = 1.0
_C.MODEL.BACKBONE_DEPTH = 1.0
_C.MODEL.BACKBONE_DEPTHWISE = False
_C.MODEL.BACKBONE_ACT = 'silu'
_C.MODEL.BACKBONE_NECT_FEATURES = ("stem", "dark2", "dark3", "dark4", "dark5")
_C.MODEL.IN_CHANNELS = [192, 384, 768]
_C.MODEL.NUM_CLASSES = 5
_C.MODEL.CONFTHRESH = 0.01
_C.MODEL.NMSTHRESH = 0.7
_C.MODEL.NUM_DUO_CONV_DPETH = 2

# -------MODEL.TRACKER-------
_C.MODEL.TRACKER = CN()
_C.MODEL.TRACKER.TRACK_THRESH = 0.7
_C.MODEL.TRACKER.FPS = 30
_C.MODEL.TRACKER.TRACK_BUFFER = 120
_C.MODEL.TRACKER.MOT20 = False
_C.MODEL.TRACKER.FIRST_MATCH_THRESH = 0.9

# -------MODEL.GRU_CVAE-------
_C.MODEL.GRU_CVAE = CN()
_C.MODEL.GRU_CVAE.INPUT_LEN = 15 # for 30 fps, 15 is 0.5 second
_C.MODEL.GRU_CVAE.PRED_LEN = 45 # for 30 fps, 15 is 0.5 second
_C.MODEL.GRU_CVAE.GLOBAL_INPUT_DIM = 4
_C.MODEL.GRU_CVAE.DEC_OUTPUT_DIM = 4
_C.MODEL.GRU_CVAE.DEC_INPUT_SIZE = 256 # the actual input size to the decoder GRU, it's the concatenation of all separate inputs
_C.MODEL.GRU_CVAE.dt = 0.4

# ----> ------FOL-------
# _C.MODEL.GRU_CVAE.IMG_SIZE = (256,256)
_C.MODEL.GRU_CVAE.ENC_CONCAT_TYPE = 'average'
_C.MODEL.GRU_CVAE.INPUT_EMBED_SIZE = 256
_C.MODEL.GRU_CVAE.ENC_HIDDEN_SIZE = 256
_C.MODEL.GRU_CVAE.DEC_HIDDEN_SIZE = 256
_C.MODEL.GRU_CVAE.GOAL_HIDDEN_SIZE = 64
_C.MODEL.GRU_CVAE.DROPOUT = 0.0
_C.MODEL.GRU_CVAE.PRIOR_DROPOUT = 0.0

# ----> ------ GOAL ------
_C.MODEL.GRU_CVAE.BEST_OF_MANY = True
_C.MODEL.GRU_CVAE.K = 20
_C.MODEL.GRU_CVAE.LATENT_DIST = 'gaussian'
_C.MODEL.GRU_CVAE.LATENT_DIM = 32 # size of Z, can be number of components of GMM when Z is categorical
_C.MODEL.GRU_CVAE.DEC_WITH_Z = True
_C.MODEL.GRU_CVAE.Z_CLIP = False
_C.MODEL.GRU_CVAE.REVERSE_LOSS = True # whether to do reverse integration to get loss
_C.MODEL.GRU_CVAE.KL_MIN = 0.07

# -------EVAL--------
_C.EVAL = CN()
_C.EVAL.BATCH_SIZE = 1
_C.EVAL.TASK = 'depth'
_C.EVAL.DATASET_NAME = 'kitti_raw'
_C.EVAL.TRAIN_JASON = "train.json"
_C.EVAL.VAL_JASON = "val_half.json"
_C.EVAL.DATASPLIT_NAME = 'train'        # 'train', 'test'
_C.EVAL.IMG_SIZE = (352, 1216)
_C.EVAL.RESIZE = (352, 1216)
_C.EVAL.SPEED_TEST = False
_C.EVAL.EVAL_FREQ = 1200
_C.EVAL.OUTPUT_DIR = '/home/azuo/LRHPerception/outputs'

# -------TRAIN-------
_C.TRAIN = CN()
_C. TRAIN.START_NEW = False
_C.TRAIN.ALL_EPOCHS = 150
_C.TRAIN.SEG_EPOCHS = 20
_C.TRAIN.DET_SEG_EPOCHS = 120
_C.TRAIN.DET_EPOCHS = 100
_C.TRAIN.DEPTH_NUM_EPOCHS = 10

_C.TRAIN.DATASET_NAME = 'kitti_raw'
# _C.TRAIN.ALL = False
_C.TRAIN.TASKS = 'depth' # depth:depth, det-seg:det-seg, pred:pred
_C.TRAIN.SEG_DATASET_NAME = 'city'
_C.TRAIN.DET_SET_LEARNING_RATE = 7e-6
_C.TRAIN.DEPTH_LEARNING_RATE = 1e-4
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.DEPTH_BATCH_SIZE = 1
_C.TRAIN.RESIZE = (352, 1216)
_C.TRAIN.TRAIN_JASON = "train.json"
_C.TRAIN.OUTPUT_DIR = '/home/azuo/LRHPerception/outputs'