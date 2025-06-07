# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import torch.nn as nn
import torch
import numpy as np
from timm.data import Mixup
from timm.utils import accuracy
from sklearn import metrics
import util.misc as misc
import util.lr_sched as lr_sched
import statistics as st
import torch.nn.functional as F
def calculate_aucs(all_labels, all_preds):

    
    all_labels = np.array(all_labels).transpose()
    all_preds =  np.array(all_preds).transpose()

    all_labels = all_labels.reshape(-1)
    all_preds = all_preds.reshape(-1)
    aucs = metrics.roc_auc_score(all_labels, all_preds,multi_class='ovr')

    return aucs

def patchify(imgs,model):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        p = model.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader_axial: Iterable, data_loader_coronal: Iterable, data_loader_sagittal: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, condition:int, topk:int,p:int,cl_ratio:float,loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 30

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.logdir))
    data_iter_step = 0
    batch_out = []
    batch_target = []
    batch_target_per = []
    y_pred = []
    loss_value = 0
    #for batch_0,batch_1,batch_2 in zip(data_loader_axial,data_loader_coronal,data_loader_sagittal):
    for data_iter_step, (samples,targets,p_prompts,s_prompts) in enumerate(metric_logger.log_every(data_loader_sagittal, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader_sagittal) + epoch, args)

        samples = F.interpolate(samples, size=(128, 128), mode='bilinear', align_corners=False)
        p_prompts = F.interpolate(p_prompts, size=(128, 128), mode='bilinear', align_corners=False)
        s_prompts = F.interpolate(s_prompts, size=(128, 128), mode='bilinear', align_corners=False)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        p_prompts = p_prompts.to(device, non_blocking=True)
        s_prompts = s_prompts.to(device, non_blocking=True)
        outputs = torch.zeros(samples.shape[0],1)
        labels = torch.zeros(samples.shape[0],1)
        for i in range(len(samples)):
            sample = samples[i].to(device)      # (T, H, W)
            p_prompt = p_prompts[i].to(device)  # (T, H, W)
            s_prompt = s_prompts[i].to(device)  # (T, H, W)
            target = targets[i, condition].unsqueeze(0).to(device)
            sample = torch.reshape(sample,(-1,1,samples.shape[-2],sample.shape[-1]))
            p_prompt = torch.reshape(p_prompt,(-1,1,p_prompt.shape[-2],p_prompt.shape[-1]))
            s_prompt = torch.reshape(s_prompt,(-1,1,s_prompt.shape[-2],s_prompt.shape[-1]))
            #out_img_list, out_s_list, out_p_list = [], [], []


            oi = model(sample)  # 去除 CLS token: (1, patch_num, C)
            os = model(s_prompt)
            op = patchify(p_prompt,model) 
            op = model.project_p(op)  # (1, patch_num, C)


    # 拼接为 [T×patch_num, C]
        
        out_img_all = oi.reshape(oi.shape[0]* oi.shape[1],oi.shape[2])
        out_s_all = os.reshape(os.shape[0]* os.shape[1],os.shape[2])
        out_p_all = op.reshape(op.shape[0]* op.shape[1],op.shape[2])

    # 拼接为 Query / Key / Value
        query = torch.cat([out_p_all, out_s_all], dim=0).unsqueeze(0)  # (1, 2×T×L, C)
        key = value = torch.cat([out_img_all.clone() for _ in range(2)], dim=0).unsqueeze(0)                         # (1, T×L, C)

        attn_out, _ = model.attn(query, key, value)  # (1, 2×T×L, C)
        print('attn_out = ',attn_out.shape)
        flat = attn_out.view( -1)

        out = model.second_fc(flat)
        output = model.last_fc(out)[0]
        outputs[i] = output
        labels[i] = target
        outputs = outputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        print('target = ',target)
        loss_cross = criterion(outputs.float(),labels.float())
        loss = loss_cross
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader_sagittal) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
        output = output.reshape(-1)
        target = target.reshape(-1)
        batch_out.append(output.cpu().detach().numpy())
        y_pred.append(output.cpu().detach().numpy())
        batch_target_per.append(target.cpu().numpy())

    auc = calculate_aucs(batch_target_per,batch_out)
    print('train auc = ',auc)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader_0,data_loader_1,data_loader_2, model, condition,device,topk,p):
#def evaluate(data_loader_0, model, device):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    batch_out = []
    batch_target = []
    batch_target_per = []
    y_pred = []
    #for batch_0,batch_1,batch_2 in zip(data_loader_0,data_loader_1,data_loader_2):
    for batch in metric_logger.log_every(data_loader_2, 10, header):
        images_0 = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        p_prompts = batch[2].to(device, non_blocking=True)
        s_prompts = batch[3].to(device, non_blocking=True)
        samples = F.interpolate(images_0, size=(128, 128), mode='bilinear', align_corners=False)
        p_prompts = F.interpolate(p_prompts, size=(128, 128), mode='bilinear', align_corners=False)
        s_prompts = F.interpolate(s_prompts, size=(128, 128), mode='bilinear', align_corners=False)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        p_prompts = p_prompts.to(device, non_blocking=True)
        s_prompts = s_prompts.to(device, non_blocking=True)
        outputs = torch.zeros(samples.shape[0],1)
        labels = torch.zeros(samples.shape[0],1)
        for i in range(len(samples)):
            sample = samples[i].to(device)      # (T, H, W)
            p_prompt = p_prompts[i].to(device)  # (T, H, W)
            s_prompt = s_prompts[i].to(device)  # (T, H, W)
            target = targets[i, condition].unsqueeze(0).to(device)
            sample = torch.reshape(sample,(-1,1,samples.shape[-2],sample.shape[-1]))
            p_prompt = torch.reshape(p_prompt,(-1,1,p_prompt.shape[-2],p_prompt.shape[-1]))
            s_prompt = torch.reshape(s_prompt,(-1,1,s_prompt.shape[-2],s_prompt.shape[-1]))
            #out_img_list, out_s_list, out_p_list = [], [], []
            oi = model(sample)  # 去除 CLS token: (1, patch_num, C)
            os = model(s_prompt)
            op = patchify(p_prompt,model)
            op = model.project_p(op)  # (1, patch_num, C)
    # 拼接为 [T×patch_num, C]

        out_img_all = oi.reshape(oi.shape[0]* oi.shape[1],oi.shape[2])
        out_s_all = os.reshape(os.shape[0]* os.shape[1],os.shape[2])
        out_p_all = op.reshape(op.shape[0]* op.shape[1],op.shape[2])

    # 拼接为 Query / Key / Value
        query = torch.cat([out_p_all, out_s_all], dim=0).unsqueeze(0)  # (1, 2×T×L, C)
        key = value = torch.cat([out_img_all.clone() for _ in range(2)], dim=0).unsqueeze(0)                         # (1, T×L, C)

        attn_out, _ = model.attn(query, key, value)  # (1, 2×T×L, C)
        print('attn_out = ',attn_out.shape)
        flat = attn_out.view( -1)

        out = model.second_fc(flat)
        output = model.last_fc(out)[0]

        target = targets[:,condition][0]
        batch_out.append(output.squeeze(0).cpu().detach().numpy())
        y_pred.append(output.squeeze(0).cpu().detach().numpy())
        batch_target_per.append(target.detach().cpu().numpy().squeeze())
    print('batch_target_per = ',batch_target_per)
    print('y_pred = ',y_pred)
    print('batch_out = ',batch_out)
    #auc = calculate_aucs(batch_out.cpu().numpy(), batch_target.cpu().numpy())
    auc = calculate_aucs(batch_target_per,batch_out)
    print('auc = ',auc)
    acc1, acc5 = accuracy(batch_out.numpy(), batch_target_per.numpy(), topk=(1, 2))
    print('acc1 = ',acc1)
    print('acc5 = ',acc5)

    metric_logger.meters['acc1'].update(auc, n=1)
    metric_logger.meters['acc3'].update(auc, n=1)
    #metric_logger.meters['auc'].update(auc.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    #print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top3.global_avg:.3f} Auc {auc.global_avg:.3f} loss {losses.global_avg:.3f}'
    #      .format(top1=metric_logger.acc1, top3=metric_logger.acc3,  auc=metric_logger.auc, losses=metric_logger.loss))
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top3.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
