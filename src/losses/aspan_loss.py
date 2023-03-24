from loguru import logger
import math

import torch
import torch.nn as nn

class ASpanLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['aspan']['loss']
        self.match_type = self.config['aspan']['match_coarse']['match_type']
        self.sparse_spvs = self.config['aspan']['match_coarse']['sparse_spvs']
        self.flow_weight=self.config['aspan']['loss']['flow_weight']

        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_flow_loss(self,coarse_corr_gt,flow_list,h0,w0,h1,w1):
        #coarse_corr_gt:[[batch_indices],[left_indices],[right_indices]]
        #flow_list: [L,B,H,W,4]
        loss1=self.flow_loss_worker(flow_list[0],coarse_corr_gt[0],coarse_corr_gt[1],coarse_corr_gt[2],w1)
        loss2=self.flow_loss_worker(flow_list[1],coarse_corr_gt[0],coarse_corr_gt[2],coarse_corr_gt[1],w0)
        total_loss=(loss1+loss2)/2
        return total_loss

    def flow_loss_worker(self,flow,batch_indicies,self_indicies,cross_indicies,w):
        bs,layer_num=flow.shape[1],flow.shape[0]
        flow=flow.view(layer_num,bs,-1,4)
        gt_flow=torch.stack([cross_indicies%w,cross_indicies//w],dim=1)

        total_loss_list=[]
        for layer_index in range(layer_num):
            cur_flow_list=flow[layer_index]
            spv_flow=cur_flow_list[batch_indicies,self_indicies][:,:2]
            spv_conf=cur_flow_list[batch_indicies,self_indicies][:,2:]#[#coarse,2]
            l2_flow_dis=((gt_flow-spv_flow)**2) #[#coarse,2]
            total_loss=(spv_conf+torch.exp(-spv_conf)*l2_flow_dis) #[#coarse,2]
            total_loss_list.append(total_loss.mean())
        total_loss=torch.stack(total_loss_list,dim=-1)*self.flow_weight
        return total_loss
        
    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                
                loss =  c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        flow_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return flow_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # print("\n\n\n\n\n\n\n\n\nCompute Fine Loss")
        # print(f"correct_mask: {correct_mask.shape}")
        # print(f"correct: {correct_mask[correct_mask == 1].shape}")
        # print(f"correct any: {correct_mask.any()}")
        # print(f"expec_f_gt: {expec_f_gt.shape}")
        # print(f"expec_f_gt abs: {expec_f_gt.abs().shape}")
        # nana = expec_f_gt[:,0].abs() > 1
        # nana = expec_f_gt[nana,1].abs() > 1
        # print(f"expec_f_gt over 1: {expec_f_gt[nana].shape}")

        # print("\n")
        # print(f"f: {expec_f.shape}")
        # print(f"f abs: {expec_f.abs().shape}")
        # print(f"f abs: {expec_f.abs()[:5,:]}")
        # nana = expec_f[:,0].abs() > 1
        # print(nana.shape)
        # nana = expec_f[nana,1].abs() > 1
        # print(nana.shape)
        # print(f"f over 1: {expec_f[nana].shape}")

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            # print("gggggggggggggggggggggggggggggggggggggggg")
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                # print("hiwuegfdhrievbiervbiehviue")
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None, torch.ones_like(expec_f[:,0], requires_grad=False)

        # l2 loss with std
        fine_error = torch.ones_like(expec_f[:,0], requires_grad=False)
        flow_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        fine_error[correct_mask] = torch.clamp(flow_l2 / (9*math.sqrt(2)/2), 0, 1)
        loss = (flow_l2 * weight[correct_mask]).mean()

        return loss, fine_error
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight)
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f, fine_err = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        data["fine_err"] = fine_err[~data["gt_mask"]].detach().cpu()
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound
        
        # 3. flow loss
        coarse_corr=[data['spv_b_ids'],data['spv_i_ids'],data['spv_j_ids']]
        loss_flow = self.compute_flow_loss(coarse_corr,data['predict_flow'],\
                                            data['hw0_c'][0],data['hw0_c'][1],data['hw1_c'][0],data['hw1_c'][1])
        loss_flow=loss_flow*self.flow_weight
        for index,loss_off in enumerate(loss_flow):
            loss_scalars.update({'loss_flow_'+str(index): loss_off.clone().detach().cpu()})  # 1 is the upper bound
            conf=data['predict_flow'][0][:,:,:,:,2:]
            layer_num=conf.shape[0]
            for layer_index in range(layer_num):
                loss_scalars.update({'conf_'+str(layer_index): conf[layer_index].mean().clone().detach().cpu()})  # 1 is the upper bound
        
        
        loss+=loss_flow.sum()
        #print((loss_c * self.loss_config['coarse_weight']).data,loss_flow.data)
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
