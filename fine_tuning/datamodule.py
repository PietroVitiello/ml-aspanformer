import os
import math
from collections import abc
from loguru import logger
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from os import path as osp
from pathlib import Path
from joblib import Parallel, delayed

import pytorch_lightning as pl
from torch import distributed as dist
from torch.utils.data import (
    Dataset,
    DataLoader,
    ConcatDataset,
    DistributedSampler,
    RandomSampler,
    dataloader,
    random_split
)

from src.utils.augment import build_augmentor
from src.utils.dataloader import get_local_split
from src.utils.misc import tqdm_joblib
from src.utils import comm
from src.datasets.megadepth import MegaDepthDataset
from src.datasets.scannet import ScanNetDataset
from src.datasets.sampler import RandomConcatSampler

from .blender_dataset import BlenderDataset


class BlenderDataModule(pl.LightningDataModule):
    """ 
    For distributed training, each training process is assgined
    only a part of the training scenes to reduce memory overhead.
    """
    def __init__(self, args, config,
                 train_split: float = 0.9996):
        super().__init__()

        # 2. dataset config
        # general options
        self.min_overlap_score_test = config.DATASET.MIN_OVERLAP_SCORE_TEST  # 0.4, omit data with overlap_score < min_overlap_score
        self.min_overlap_score_train = config.DATASET.MIN_OVERLAP_SCORE_TRAIN
        self.augment_fn = build_augmentor(config.DATASET.AUGMENTATION_TYPE)  # None, options: [None, 'dark', 'mobile']

        # MegaDepth options
        self.mgdpt_img_resize = config.DATASET.MGDPT_IMG_RESIZE  # 840
        self.mgdpt_img_pad = config.DATASET.MGDPT_IMG_PAD   # True
        self.mgdpt_depth_pad = config.DATASET.MGDPT_DEPTH_PAD   # True
        self.mgdpt_df = config.DATASET.MGDPT_DF  # 8
        self.coarse_scale = 1 / config.ASPAN.RESOLUTION[0]  # 0.125. for training loftr.

        # 3.loader parameters
        self.train_loader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        self.val_loader_params = {
            'batch_size': args.batch_size_val,
            'shuffle': False,
            'num_workers': args.num_workers,
            'pin_memory': getattr(args, 'pin_memory', True)
        }
        
        self.n_samples_per_subset = config.TRAINER.N_SAMPLES_PER_SUBSET
        self.subset_replacement = config.TRAINER.SB_SUBSET_SAMPLE_REPLACEMENT
        self.shuffle = config.TRAINER.SB_SUBSET_SHUFFLE
        self.repeat = config.TRAINER.SB_REPEAT
        
        # (optional) RandomSampler for debugging

        # misc configurations
        # self.parallel_load_data = getattr(args, 'parallel_load_data', False)
        self.seed = config.TRAINER.SEED  # 66

        self.train_split = train_split
        self.full_dataset = BlenderDataset(args.use_masks, args.crop_margin, args.resize_modality,
                                           args.segment_object, args.filter_dataset)
        self.training_dataset, self.validation_dataset = None, None

    def setup(self, stage=None):
        """
        Setup train / val / test dataset. This method will be called by PL automatically.
        Args:
            stage (str): 'fit' in training phase, and 'test' in testing phase.
        """

        assert stage in ['fit', 'test'], "stage must be either fit or test"

        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            logger.info(f"[rank:{self.rank}] world_size: {self.world_size}")
        except AssertionError as ae:
            self.world_size = 1
            self.rank = 0
            logger.warning(str(ae) + " (set wolrd_size=1 and rank=0)")

        train_samples = round(self.train_split * len(self.full_dataset))
        val_samples = len(self.full_dataset) - train_samples
        self.training_dataset, self.validation_dataset = random_split(
            self.full_dataset, [train_samples, val_samples]
        )
        logger.info(f'[rank:{self.rank}] Train & Val Dataset loaded!')

    def train_dataloader(self):
        """ Build training dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Train Sampler and DataLoader re-init (should not re-init between epochs!).')
        return DataLoader(self.training_dataset, **self.train_loader_params)
    
    def val_dataloader(self):
        """ Build validation dataloader for ScanNet / MegaDepth. """
        logger.info(f'[rank:{self.rank}/{self.world_size}]: Val Sampler and DataLoader re-init.')
        return DataLoader(self.validation_dataset, **self.val_loader_params)
