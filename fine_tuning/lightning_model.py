from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path
import wandb

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.ASpanFormer.aspanformer import ASpanFormer
from src.ASpanFormer.utils.supervision import compute_supervision_coarse, compute_supervision_fine
from src.losses.aspan_loss import ASpanLoss
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_symmetrical_epipolar_errors,compute_symmetrical_epipolar_errors_offset_bidirectional,
    compute_pose_errors,
    aggregate_metrics
)
from src.utils.plotting import make_matching_figures,make_matching_figures_offset
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler


class PL_ASpanFormer(pl.LightningModule):
    def __init__(self,
                 config,
                 pretrained_ckpt=None,
                 profiler=None,
                 dump_dir=None,
                 use_wandb: bool = True):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        _config = lower_config(self.config)
        self.loftr_cfg = lower_config(_config['aspan'])
        self.profiler = profiler or PassThroughProfiler()
        self.n_vals_plot = 2 #max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)

        # Matcher: LoFTR
        self.matcher = ASpanFormer(config=_config['aspan'])
        self.loss = ASpanLoss(_config)
        self.optimizer = None # Will be set later in a lightning function

        # Pretrained weights
        print(pretrained_ckpt)
        if pretrained_ckpt:
            print('load')
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            msg=self.matcher.load_state_dict(state_dict, strict=False)
            print(msg)
            logger.info(f"Load \'{pretrained_ckpt}\' as pretrained checkpoint")
        
        # Testing
        self.dump_dir = dump_dir

        self.use_wandb = use_wandb
        if self.global_rank == 0 and use_wandb:
            entity = "head-dome"
            print(f'Connecting with {entity} on wandb')
            wandb.init(
                project="headcam-dome",
                name=config.MODEL.NAME,
                entity=entity,
                reinit=True,
                tags=["ASpanFormer"]
            )
            wandb.config = {
                "learning_rate": config.TRAINER.TRUE_LR,
                "epochs": config.TRAINER.MAX_EPOCHS,
                "batch_size": config.TRAINER.TRUE_BATCH_SIZE,
                "using_segmask": config.MODEL.MASK,
                "resize_modality": config.MODEL.RESIZE,
            }
            wandb.watch(self.matcher, log='all', log_freq=1)

        self.train_step = 0
        self.val_step = 0
        self.train_loss = np.array([np.inf]) 

    def wandb_log_epochs(self, data: dict, stage: str='train'):
        if stage == 'train':
            loss_data = data["loss_scalars"]
            flow_losses = {}
            for loss_key in [key for key in loss_data.keys() if key.startswith("loss_flow_")]:
                id = loss_key.split('_')[-1]
                flow_losses.update({f"train_flow_loss_{id}": loss_data[loss_key]})
            wandb.log({"epoch": self.current_epoch,
                       "train_step": self.train_step,
                       "train_loss_stepwise": self.train_loss[-1],
                       "training_window_loss": np.mean(self.train_loss[1:]),
                       "train_coarse_loss": loss_data["loss_c"],
                       "train_fine_loss": loss_data["loss_f"],
                       **flow_losses,
                       'train_figures': data["train_figures"],
                       "lr": self.optimizer.param_groups[0]["lr"],
                    })
            self.train_step += 1
            
        elif stage == 'val':
            data.update({
                "epoch": self.current_epoch,
                "val_step" : self.val_step,
                "train_loss": np.mean(self.train_loss[1:]),
            })
            wandb.log(data)
            self.val_step += 1
        else:
            print("[WANDB] Incorrect logging stage")
        data.clear()
        
    def configure_optimizers(self):
        # FIXME: The scheduler did not work properly when `--resume_from_checkpoint`
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        self.optimizer = optimizer
        return [optimizer], [scheduler]
    
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx,
            optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # learning rate warm up
        warmup_step = self.config.TRAINER.WARMUP_STEP
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                    (self.trainer.global_step / self.config.TRAINER.WARMUP_STEP) * \
                    abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                pass
            else:
                raise ValueError(f'Unknown lr warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        # update lr stepwise
        scheduler = self.lr_schedulers()
        scheduler.step()
    
    def _trainval_inference(self, batch):
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config) 
        
        with self.profiler.profile("LoFTR"):
            self.matcher(batch) 
        
        with self.profiler.profile("Compute fine supervision"):
            compute_supervision_fine(batch, self.config) 
        
        with self.profiler.profile("Compute losses"):
            self.loss(batch) 
    
    def _compute_metrics(self, batch):
        with self.profiler.profile("Copmute metrics"):
            compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
            compute_symmetrical_epipolar_errors_offset_bidirectional(batch) # compute epi_errs for offset match
            compute_pose_errors(batch, self.config)  # compute R_errs, t_errs, pose_errs for each pair

            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['image0'].size(0)
            metrics = {
                # to filter duplicate pairs caused by DistributedSampler
                'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
                'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
                'epi_errs_offset': [batch['epi_errs_offset_left'][batch['offset_bids_left'] == b].cpu().numpy() for b in range(bs)], #only consider left side
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']}
            ret_dict = {'metrics': metrics}
        return ret_dict, rel_pair_names
    
    def on_train_epoch_start(self) -> None:
        self.train_loss = np.zeros(1)
   
    def training_step(self, batch, batch_idx):
        self._trainval_inference(batch)
        self.train_loss = np.append(self.train_loss, batch['loss'].detach().cpu().numpy())
        
        # logging
        if self.global_step % self.trainer.log_every_n_steps == 0:
            # # scalars
            # for k, v in batch['loss_scalars'].items():
            #     if not k.startswith('loss_flow') and not k.startswith('conf_'):
            #         self.logger.experiment.add_scalar(f'train/{k}', v, self.global_step)
            
            # #log offset_loss and conf for each layer and level
            # layer_num=self.loftr_cfg['coarse']['layer_num']
            # for layer_index in range(layer_num):
            #     log_title='layer_'+str(layer_index)
            #     self.logger.experiment.add_scalar(log_title+'/offset_loss', batch['loss_scalars']['loss_flow_'+str(layer_index)], self.global_step)
            #     self.logger.experiment.add_scalar(log_title+'/conf_', batch['loss_scalars']['conf_'+str(layer_index)],self.global_step)
            
            # # net-params
            # if self.config.ASPAN.MATCH_COARSE.MATCH_TYPE == 'sinkhorn':
            #     self.logger.experiment.add_scalar(
            #         f'skh_bin_score', self.matcher.coarse_matching.bin_score.clone().detach().cpu().data, self.global_step)

            window_average_loss = np.mean(self.train_loss[1:])
            self.log("training_window_loss", window_average_loss)
            self.log("training_last_loss", self.train_loss[-1])

            if self.trainer.global_rank == 0:
                if self.use_wandb:
                    # figures
                    train_figures = []
                    if self.config.TRAINER.ENABLE_PLOTTING:
                        # compute_symmetrical_epipolar_errors(batch)  # compute epi_errs for each match
                        figures = make_matching_figures(batch, self.config, self.config.TRAINER.PLOT_MODE)
                        for k, v in figures.items():
                            # self.logger.experiment.add_figure(f'train_match/{k}', v, self.global_step)
                            for plot_idx, fig in enumerate(v):
                                fig.canvas.draw()
                                w, h = fig.canvas.get_width_height()
                                fig = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
                                fig.shape = (h,w,4)
                                fig = torch.from_numpy(fig[:,:,1:].transpose(2,0,1)).float()
                                train_figures.append(wandb.Image(fig, caption=f"Figure {k}_{plot_idx}"))
                    #plot offset 
                    # if self.global_step%200==0:
                    #     compute_symmetrical_epipolar_errors_offset_bidirectional(batch)
                    #     figures_left = make_matching_figures_offset(batch, self.config, self.config.TRAINER.PLOT_MODE,side='_left')
                    #     figures_right = make_matching_figures_offset(batch, self.config, self.config.TRAINER.PLOT_MODE,side='_right')
                    #     for k, v in figures_left.items():
                    #         self.logger.experiment.add_figure(f'train_offset/{k}'+'_left', v, self.global_step)
                    #     figures = make_matching_figures_offset(batch, self.config, self.config.TRAINER.PLOT_MODE,side='_right')
                    #     for k, v in figures_right.items():
                    #         self.logger.experiment.add_figure(f'train_offset/{k}'+'_right', v, self.global_step)

                    train_data = {
                        **batch,
                        'train_figures': train_figures
                    }
                    self.wandb_log_epochs(train_data)

                self.train_loss = self.train_loss[0]
                plt.close('all')
                
        return {'loss': batch['loss']}
    
    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx: int) -> None:
        outputs.clear()
        
    # def training_epoch_end(self, outputs):
    #     # avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    #     # if self.trainer.global_rank == 0:
    #     #     self.logger.experiment.add_scalar(
    #     #         'train/avg_loss_on_epoch', avg_loss,
    #     #         global_step=self.current_epoch)
    #     # self.train_epoch_loss = outputs[-1]["loss"].clone().detach().cpu()
    
    def validation_step(self, batch, batch_idx):
        self._trainval_inference(batch)
         
        ret_dict, _ = self._compute_metrics(batch) #this func also compute the epi_errors
        
        val_plot_interval = max(self.trainer.num_val_batches[0] // self.n_vals_plot, 1)
        figures = {self.config.TRAINER.PLOT_MODE: []}
        figures_offset = {self.config.TRAINER.PLOT_MODE: []}
        if self.trainer.global_rank == 0:
            if batch_idx % val_plot_interval == 0:
                figures = make_matching_figures(batch, self.config, mode=self.config.TRAINER.PLOT_MODE)
                figures_offset=make_matching_figures_offset(batch, self.config, self.config.TRAINER.PLOT_MODE,'_left')

        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures,
            'figures_offset_left':figures_offset
        }
        
    def validation_epoch_end(self, outputs):
        # handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        val_data = {}
        figures_list = []
        for valset_idx, outputs in enumerate(multi_outputs):
            # since pl performs sanity_check at the very begining of the training
            # cur_epoch = self.trainer.current_epoch
            # if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
            #     cur_epoch = -1

            # 1. loss_scalars: dict of list, on cpu
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) for k in _loss_scalars[0]}

            # 2. val metrics: dict of list, numpy
            _metrics = [o['metrics'] for o in outputs]
            metrics = {k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
            # NOTE: all ranks need to `aggregate_merics`, but only log at rank-0 
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            
            # 3. figures
            _figures = [o['figures'] for o in outputs]
            figures = {k: flattenList(gather(flattenList([_me[k] for _me in _figures]))) for k in _figures[0]}

            # tensorboard records only on rank 0
            if self.trainer.global_rank == 0:
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    # self.logger.experiment.add_scalar(f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch)
                    val_data.setdefault(f"val_avg_{k}", []).append(mean_v)
                    # val_data.update({f"val_avg_{k}": mean_v})

                for k, v in val_metrics_4tb.items():
                    # self.logger.experiment.add_scalar(f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch)
                    val_data.setdefault(f"val_metric_{k}", []).append(v)
                    # val_data.update({f"val_metric_{k}": v})
                
                for k, v in figures.items():
                    for plot_idx, fig in enumerate(v):
                        # self.logger.experiment.add_figure(
                        #     f'val_match_{valset_idx}/{k}/pair-{plot_idx}', fig, cur_epoch, close=True)
                        # plt.show()
                        if self.use_wandb:
                            fig.canvas.draw()
                            w, h = fig.canvas.get_width_height()
                            fig = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
                            fig.shape = (h,w,4)
                            fig = torch.from_numpy(fig[:,:,1:].transpose(2,0,1)).float()
                            figures_list.append(wandb.Image(fig, caption=f"Figure {k}_{plot_idx}"))

        for key in val_data.keys():
            info = np.array(val_data[key])
            val_data[key] = np.mean(info)

        val_data.update({
            "val_figures": figures_list
        })

        self.log("val_loss", val_data["val_avg_loss"])
                            
        if self.trainer.global_rank == 0 and self.use_wandb:
            self.wandb_log_epochs(val_data, stage='val')
        plt.close('all')

        for thr in [5, 10, 20]:
            # log on all ranks for ModelCheckpoint callback to work properly
            self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))  # ckpt monitors on this

    def test_step(self, batch, batch_idx):
        with self.profiler.profile("LoFTR"):
            self.matcher(batch)

        ret_dict, rel_pair_names = self._compute_metrics(batch)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                # dump results for further analysis
                keys_to_save = {'mkpts0_f', 'mkpts1_f', 'mconf', 'epi_errs'}
                pair_names = list(zip(*batch['pair_names']))
                bs = batch['image0'].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch['m_bids'] == b_id
                    item['pair_names'] = pair_names[b_id]
                    item['identifier'] = '#'.join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        item[key] = batch[key][mask].cpu().numpy()
                    for key in ['R_errs', 't_errs', 'inliers']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict['dumps'] = dumps

        return ret_dict

    def test_epoch_end(self, outputs):
        # metrics: dict of list, numpy
        _metrics = [o['metrics'] for o in outputs]
        metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])  # [{...}, #bs*#batch]
            dumps = flattenList(gather(_dumps))  # [{...}, #proc*#bs*#batch]
            logger.info(f'Prediction and evaluation results will be saved to: {self.dump_dir}')

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / 'LoFTR_pred_eval', dumps)
