import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
# from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_aspanformer import PL_ASpanFormer
from fine_tuning.preprocessing import get_resize_modality_name

from fine_tuning.datamodule import BlenderDataModule

loguru_logger = get_rank_zero_only_logger(loguru_logger)

def parse_args():
    def str2bool(v: str) -> bool:
        return v.lower() in ("true", "1")
    # init a custom parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-name', '--model_name', type=str, default=None, required=True,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '-resize_m', '--resize_modality', type=int, default=0, required=False,
        help='Set the modality used to resize the images. Options: [0-5]')
    parser.add_argument(
        '-mask', '--use_masks', action='store_true',
        help='Whether to upload the training information to weights and biases')
    parser.add_argument(
        '-wandb', '--use_wandb', action='store_true',
        help='Whether to upload the training information to weights and biases')
    parser.add_argument(
        '-bs', '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '-bs_val', '--batch_size_val', type=int, default=None, help='validation set batch_size per gpu')
    parser.add_argument(
        '-nw', '--num_workers', type=int, default=0)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=False, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--exp_name', type=str, default='trying_out')
    
    parser.add_argument(
        '--data_cfg_path', type=str, help='data config path', default="configs/data/scannet_trainval.py")
    parser.add_argument(
        '--main_cfg_path', type=str, help='main config path', default="configs/aspan/indoor/aspan_train.py")
    parser.add_argument(
        '--ckpt_path', type=str, default=Path("weights/indoor.ckpt"),
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')
    
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')    
    parser.add_argument(
        '--mode', type=str, default='vanilla',
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')
    parser.add_argument(
        '--ini', type=str2bool, default=False,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')

    parser = pl.Trainer.add_argparse_args(parser)
    '''
    Useful Trainer arguments:
        - log_every_n_steps
        - max_epochs
        - gpus
    '''
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()
    # rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(
        config.TRAINER.WARMUP_STEP / _scaling)
    
    config.MODEL.NAME = args.model_name
    config.MODEL.MASK = args.use_masks
    config.MODEL.RESIZE = get_resize_modality_name(args.resize_modality)
    config.TRAINER.MAX_EPOCHS = args.max_epochs
    if args.batch_size_val is None:
        args.batch_size_val = args.batch_size

    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_ASpanFormer(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, use_wandb=args.use_wandb)
    loguru_logger.info(f"ASpanFormer LightningModule initialized!")

    # lightning data
    # data_module = MultiSceneDataModule(args, config)
    data_module = BlenderDataModule(args, config)
    loguru_logger.info(f"ASpanFormer DataModule initialized!")

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'

    # Callbacks
    # TODO: update ModelCheckpoint to monitor multiple metrics
    ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=5, mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)

    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=DDPPlugin(find_unused_parameters=False,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()




























# def parse_args():
#     def str2bool(v: str) -> bool:
#         return v.lower() in ("true", "1")
#     # init a custom parser which will be added into pl.Trainer parser
#     # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         'data_cfg_path', type=str, help='data config path')
#     parser.add_argument(
#         'main_cfg_path', type=str, help='main config path')
#     parser.add_argument(
#         '--exp_name', type=str, default='default_exp_name')
#     parser.add_argument(
#         '--batch_size', type=int, default=4, help='batch_size per gpu')
#     parser.add_argument(
#         '--num_workers', type=int, default=4)
#     parser.add_argument(
#         '--pin_memory', type=lambda x: bool(strtobool(x)),
#         nargs='?', default=True, help='whether loading data to pinned memory or not')
#     parser.add_argument(
#         '--ckpt_path', type=str, default=None,
#         help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')
#     parser.add_argument(
#         '--disable_ckpt', action='store_true',
#         help='disable checkpoint saving (useful for debugging).')
#     parser.add_argument(
#         '--profiler_name', type=str, default=None,
#         help='options: [inference, pytorch], or leave it unset')
#     parser.add_argument(
#         '--parallel_load_data', action='store_true',
#         help='load datasets in with multiple processes.')
#     parser.add_argument(
#         '--mode', type=str, default='vanilla',
#         help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')
#     parser.add_argument(
#         '--ini', type=str2bool, default=False,
#         help='pretrained checkpoint path, helpful for using a pre-trained coarse-only ASpanFormer')

#     parser = pl.Trainer.add_argparse_args(parser)
#     return parser.parse_args()


# def main():
#     # parse arguments
#     args = parse_args()
#     rank_zero_only(pprint.pprint)(vars(args))

#     # init default-cfg and merge it with the main- and data-cfg
#     config = get_cfg_defaults()
#     config.merge_from_file(args.main_cfg_path)
#     config.merge_from_file(args.data_cfg_path)
#     pl.seed_everything(config.TRAINER.SEED)  # reproducibility
#     # TODO: Use different seeds for each dataloader workers
#     # This is needed for data augmentation

#     # scale lr and warmup-step automatically
#     args.gpus = _n_gpus = setup_gpus(args.gpus)
#     config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
#     config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
#     _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
#     config.TRAINER.SCALING = _scaling
#     config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
#     config.TRAINER.WARMUP_STEP = math.floor(
#         config.TRAINER.WARMUP_STEP / _scaling)

#     # lightning module
#     profiler = build_profiler(args.profiler_name)
#     model = PL_ASpanFormer(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
#     loguru_logger.info(f"ASpanFormer LightningModule initialized!")

#     # lightning data
#     data_module = MultiSceneDataModule(args, config)
#     loguru_logger.info(f"ASpanFormer DataModule initialized!")

#     # TensorBoard Logger
#     logger = TensorBoardLogger(
#         save_dir='logs/tb_logs', name=args.exp_name, default_hp_metric=False)
#     ckpt_dir = Path(logger.log_dir) / 'checkpoints'

#     # Callbacks
#     # TODO: update ModelCheckpoint to monitor multiple metrics
#     ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=5, mode='max',
#                                     save_last=True,
#                                     dirpath=str(ckpt_dir),
#                                     filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
#     lr_monitor = LearningRateMonitor(logging_interval='step')
#     callbacks = [lr_monitor]
#     if not args.disable_ckpt:
#         callbacks.append(ckpt_callback)

#     # Lightning Trainer
#     trainer = pl.Trainer.from_argparse_args(
#         args,
#         plugins=DDPPlugin(find_unused_parameters=False,
#                           num_nodes=args.num_nodes,
#                           sync_batchnorm=config.TRAINER.WORLD_SIZE > 0),
#         gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
#         callbacks=callbacks,
#         logger=logger,
#         sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
#         replace_sampler_ddp=False,  # use custom sampler
#         reload_dataloaders_every_epoch=False,  # avoid repeated samples!
#         weights_summary='full',
#         profiler=profiler)
#     loguru_logger.info(f"Trainer initialized!")
#     loguru_logger.info(f"Start training!")
#     trainer.fit(model, datamodule=data_module)


# if __name__ == '__main__':
#     main()



# import pytorch_lightning as pl
# import argparse
# import pprint
# from loguru import logger as loguru_logger

# from src.config.default import get_cfg_defaults
# from src.utils.profiler import build_profiler

# from src.lightning.data import MultiSceneDataModule
# from src.lightning.lightning_aspanformer import PL_ASpanFormer
# import torch

# def parse_args():
#     # init a costum parser which will be added into pl.Trainer parser
#     # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument(
#         'data_cfg_path', type=str, help='data config path')
#     parser.add_argument(
#         'main_cfg_path', type=str, help='main config path')
#     parser.add_argument(
#         '--ckpt_path', type=str, default="weights/indoor_ds.ckpt", help='path to the checkpoint')
#     parser.add_argument(
#         '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
#     parser.add_argument(
#         '--profiler_name', type=str, default=None, help='options: [inference, pytorch], or leave it unset')
#     parser.add_argument(
#         '--batch_size', type=int, default=1, help='batch_size per gpu')
#     parser.add_argument(
#         '--num_workers', type=int, default=2)
#     parser.add_argument(
#         '--thr', type=float, default=None, help='modify the coarse-level matching threshold.')
#     parser.add_argument(
#         '--mode', type=str, default='vanilla', help='modify the coarse-level matching threshold.')
#     parser = pl.Trainer.add_argparse_args(parser)
#     return parser.parse_args()


# if __name__ == '__main__':
#     # parse arguments
#     args = parse_args()
#     pprint.pprint(vars(args))

#     # init default-cfg and merge it with the main- and data-cfg
#     config = get_cfg_defaults()
#     config.merge_from_file(args.main_cfg_path)
#     config.merge_from_file(args.data_cfg_path)
#     pl.seed_everything(config.TRAINER.SEED)  # reproducibility

#     # tune when testing
#     if args.thr is not None:
#         config.ASPAN.MATCH_COARSE.THR = args.thr

#     loguru_logger.info(f"Args and config initialized!")

#     # lightning module
#     profiler = build_profiler(args.profiler_name)
#     model = PL_ASpanFormer(config, pretrained_ckpt=args.ckpt_path, profiler=profiler, dump_dir=args.dump_dir)
#     loguru_logger.info(f"ASpanFormer-lightning initialized!")

#     # lightning data
#     data_module = MultiSceneDataModule(args, config)
#     loguru_logger.info(f"DataModule initialized!")

#     # lightning trainer
#     trainer = pl.Trainer.from_argparse_args(args, replace_sampler_ddp=False, logger=False)

#     loguru_logger.info(f"Start testing!")
#     trainer.test(model, datamodule=data_module, verbose=False)

