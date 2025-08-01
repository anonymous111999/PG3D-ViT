# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

#assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit

from engine_finetune import train_one_epoch, evaluate
from src.data_loader import make_data_loader_finetune
from src.utils import create_output_dir, \
                  print_stats,       \
                  save_losses,       \
                  save_checkpoint

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--condition', default=0, type=int)
    parser.add_argument('--topk', default=15000, type=int)
    parser.add_argument('--cl_ratio', default=0.1, type=float)
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--sample_hard', default=False, type=bool,
                        help='whether focus on hard')
    parser.add_argument('--sample_hard_level', default=0, type=float,
                        help='level less,double more')
    parser.add_argument('--ssim_ratio', default=0, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False,type=bool,
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    args.distributed = False
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    # simple augmentation
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_train_axial = make_data_loader_finetune(args.data_path, 'train', 'axial',args.condition)
    dataset_val_axial = make_data_loader_finetune(args.data_path, 'valid', 'axial',args.condition)
    dataset_train_coronal = make_data_loader_finetune(args.data_path, 'train', 'coronal',args.condition)
    dataset_val_coronal = make_data_loader_finetune(args.data_path, 'valid', 'coronal',args.condition)
    dataset_train_sagittal = make_data_loader_finetune(args.data_path, 'train', 'sagittal',args.condition)
    dataset_val_sagittal = make_data_loader_finetune(args.data_path, 'valid', 'sagittal',args.condition)

    #if True:  # args.distributed:
    if False:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train_axial = torch.utils.data.DistributedSampler(
            dataset_train_axial, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_train_coronal = torch.utils.data.DistributedSampler(
            dataset_train_coronal, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_train_sagittal = torch.utils.data.DistributedSampler(
            dataset_train_sagittal, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        if args.dist_eval:
            if len(dataset_val_axial) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val_axial = torch.utils.data.DistributedSampler(
                dataset_val_axial, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
            sampler_val_coronal = torch.utils.data.DistributedSampler(
                dataset_val_coronal, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
            sampler_val_sagittal = torch.utils.data.DistributedSampler(
                dataset_val_sagittal, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
        else:
            sampler_val_axial= torch.utils.data.SequentialSampler(dataset_val_axial)
            sampler_val_coronal= torch.utils.data.SequentialSampler(dataset_val_coronal)
            sampler_val_sagittal= torch.utils.data.SequentialSampler(dataset_val_sagittal)
    else:
        sampler_train_axial = torch.utils.data.RandomSampler(dataset_train_axial)
        sampler_val_axial = torch.utils.data.SequentialSampler(dataset_val_axial)
        sampler_train_coronal = torch.utils.data.RandomSampler(dataset_train_coronal)
        sampler_val_coronal = torch.utils.data.SequentialSampler(dataset_val_coronal)
        sampler_train_sagittal = torch.utils.data.RandomSampler(dataset_train_sagittal)
        sampler_val_sagittal = torch.utils.data.SequentialSampler(dataset_val_sagittal)
    global_rank = 0
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_axial = torch.utils.data.DataLoader(
        dataset_train_axial, sampler=sampler_train_axial,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val_axial = torch.utils.data.DataLoader(
        dataset_val_axial, sampler=sampler_val_axial,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_train_coronal = torch.utils.data.DataLoader(
        dataset_train_coronal, sampler=sampler_train_coronal,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val_coronal = torch.utils.data.DataLoader(
        dataset_val_coronal, sampler=sampler_val_coronal,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    data_loader_train_sagittal = torch.utils.data.DataLoader(
        dataset_train_sagittal, sampler=sampler_train_sagittal,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val_sagittal = torch.utils.data.DataLoader(
        dataset_val_sagittal, sampler=sampler_val_sagittal,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        #if args.global_pool:
        ##    #assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
         ##   expected_missing = {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        ##    assert expected_missing.issubset(set(msg.missing_keys))
        ##else:
         ##   assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    emb_model_48 = ['vit_base_patch4','vit_base_patch8']
    emb_model_96 = ['vit_huge_patch4','vit_huge_patch8']
    emb_model_24 = ['vit_small_patch4','vit_small_patch8']
    patch_4 = ['vit_base_patch4','vit_huge_patch4','vit_small_patch4']
    patch_8 = ['vit_base_patch8','vit_huge_patch8','vit_small_patch8']
    emb_dim = 48
    patch_num = 1025
    #if args.model in patch_8:
    #    patch_num = 256
    if args.model in emb_model_48:
        emb_dim = 48
    if args.model in emb_model_96:
        emb_dim = 96
    if args.model in emb_model_24:
        emb_dim = 24
    slice_num = 8
    model.project_p = nn.Linear(16,96)
    model.attn = nn.MultiheadAttention(embed_dim=96, num_heads=1, batch_first=True)
    model.second_fc = nn.Linear(emb_dim*16*(2*patch_num-1),16)
    model.last_fc = nn.Linear(16,2)
    if args.warmup_epochs !=0:
        print('model rawm up from ', args.warmup_epochs)
        checkpoint = torch.load(str(args.output_dir) + '/checkpoint-'+ str(args.warmup_epochs) + '.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 25

    print("base lr: %.2e" % (args.lr * 25 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    #if args.distributed:
    if False:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    #if mixup_fn is not None:
        # smoothing is handled with mixup label transform
    #    criterion = SoftTargetCrossEntropy()
    #elif args.smoothing > 0.:
    #    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    #else:
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    if args.eval:
        test_stats = evaluate(data_loader_val_axial, data_loader_val_coronal,data_loader_val_sagittal,model, args.condition,device)
        print(f"Accuracy of the network on the {len(dataset_val_axial)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    p=8
    for epoch in range(args.warmup_epochs, args.epochs):
        #if args.distributed:
        if False:
            data_loader_train_axial.sampler.set_epoch(epoch)
            data_loader_train_coronal.sampler.set_epoch(epoch)
            data_loader_train_sagittal.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train_axial,data_loader_train_coronal,data_loader_train_sagittal,
            optimizer, device, epoch,args.condition, args.topk,p,args.cl_ratio,loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        test_stats = evaluate(data_loader_val_axial, data_loader_val_coronal,data_loader_val_sagittal, model, args.condition, device,args.topk,p)
        print(f"Accuracy of the network on the {len(dataset_val_axial)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc3', test_stats['acc3'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
