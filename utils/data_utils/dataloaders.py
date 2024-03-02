import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from .datasets import kitti_road_dataset, kitti_det_dataset

class MOT_Dataloaders():
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.dataroot = cfgs.DATASET.ROOT
        
    def get_data_loader(self, is_distributed, no_aug=False):
        from . import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )
        batch_size = self.cfgs.TRAIN.BATCH_SIZE

        dataset = MOTDataset(
            data_dir=os.path.join(self.dataroot, self.cfgs.TRAIN.DATASET_NAME),
            json_file=self.args.TRAIN.TRAIN_JASON,
            name='',
            img_size=self.args.DATALOADER.RESIZE,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
    
    def get_eval_loader(self, is_distributed):
        from . import ValTransform
        from .datasets import MOTDataset

        batch_size = self.cfgs.EVAL.BATCH_SIZE

        valdataset = MOTDataset(
            self.cfgs,
            data_dir=os.path.join(self.dataroot, self.cfgs.EVAL.DATASET_NAME),
            json_file=self.cfgs.EVAL.VAL_JASON,
            img_size=self.cfgs.DATALOADER.RESIZE,
            name=self.cfgs.EVAL.DATASPLIT_NAME,   # change to train when running on training set
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
        
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.cfgs.DATALOADER.NUM_WORKERS,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader
  