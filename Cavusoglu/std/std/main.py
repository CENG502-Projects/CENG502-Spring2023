# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Copyright 2020 - present, Facebook, Inc
# Copyright 2023 Devrim Cavusoglu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Main script to control different procedures (training, evaluation, etc.).
This file is taken and adapted for STD implementation from Facebook
Research DeiT repository. See the original source below
https://github.com/facebookresearch/deit/blob/main/main.py
"""
import argparse
import datetime
import json
import time
from contextlib import suppress
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler

import std.utils as utils
from std import NEPTUNE_CONFIG_PATH
from std.args import get_args_parser
from std.dataset import build_dataset
from std.engine import evaluate, train_one_epoch
from std.losses import DistillationLoss
from std.mine import build_mine, mine_regularization
from std.models.mlp_mixer import MLPMixer
from std.models.std_mlp_mixer import STDMLPMixer
from std.samplers import RASampler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    from timm.utils import ApexScaler

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from fvcore.nn import FlopCountAnalysis, flop_count, flop_count_table, parameter_count
    from utils import sfc_flop_jit

    has_fvcore = True
except ImportError:
    has_fvcore = False


def set_seed(seed):
    seed = seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)


def create_criterion(args):
    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    return criterion


def create_loaders(dataset_train, dataset_val, distributed=True):
    if distributed:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return data_loader_train, data_loader_val


def get_model(args):
    if "mlp-mixer" in args.model:
        if "std" in args.model:
            return STDMLPMixer(
                image_size=args.input_size,
                channels=args.channels,
                patch_size=args.patch_size,
                dim=args.embedding_dim,
                depth=args.depth,
                dropout=args.drop,
                num_classes=args.nb_classes,
                distill_intermediate=args.distill_intermediate,
                n_teachers=len(args.teacher_model),
            )
        else:
            return MLPMixer(
                image_size=args.input_size,
                channels=args.channels,
                patch_size=args.patch_size,
                dim=args.embedding_dim,
                depth=args.depth,
                dropout=args.drop,
                num_classes=args.nb_classes,
            )
    else:
        print("Only MLP-Mixer is available currently, others will come soon!")


def train(
    model,
    data_loader_train,
    data_loader_val,
    criterion,
    optimizer,
    loss_scaler,
    neptune_run: Optional = None,
    *,
    model_ema,
    mixup_fn,
    amp_autocast,
    lr_scheduler,
    output_dir,
    model_without_ddp,
    dataset_val,
    n_parameters,
    args,
    device,
):
    # build MINE stuff
    dim_spatial = args.embedding_dim
    dim_channel = (args.input_size // args.patch_size) ** 2
    if args.distillation_type != "none":
        model_regulizer, mine_network, mine_optimizer, objective = build_mine(
            model, dim_spatial, dim_channel, device
        )

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, mine_samples = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.distillation_type,
            args.clip_grad,
            model_ema,
            mixup_fn,
            set_training_mode=args.finetune == "",  # keep in eval mode during finetuning
            amp_autocast=amp_autocast,
            n_mine_samples=args.n_mine_samples,
            neptune_run=neptune_run,
        )
        # regularize with MINE
        if mine_samples is not None:
            mine_regularization(
                model,
                mine_network,
                model_regulizer,
                mine_optimizer,
                objective,
                mine_samples,
                neptune_run=neptune_run,
            )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        # "model_ema": get_state_dict(model_ema),
                        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate(
            data_loader_val, model, device, amp_autocast=amp_autocast, neptune_run=neptune_run
        )
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def prepare_for_finetune(model):
    if args.finetune.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.finetune, map_location="cpu", check_hash=True
        )
    else:
        checkpoint = torch.load(args.finetune, map_location="cpu")

    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias", "head_dist.weight", "head_dist.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
    )
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model["pos_embed"] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)
    return model


def main(args):
    neptune_run = None
    if args.log_neptune:
        neptune_run = utils.create_experiment(config_path=NEPTUNE_CONFIG_PATH)
        neptune_run["arguments"] = args
    utils.init_distributed_mode(args)

    print(args)

    if args.distillation_type != "none" and args.finetune and not args.eval:
        raise NotImplementedError("Finetuning with distillation not yet supported")

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if not args.no_amp:  # args.amp: Default  use AMP
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
            args.apex_amp = False
        elif has_apex:
            args.native_amp = False
            args.apex_amp = True
        else:
            raise ValueError(
                "Warning: Neither APEX or native Torch AMP is available, using float32."
                "Install NVIDA apex or upgrade to PyTorch 1.6"
            )
    else:
        args.apex_amp = False
        args.native_amp = False
    if args.apex_amp and has_apex:
        use_amp = "apex"
    elif args.native_amp and has_native_amp:
        use_amp = "native"
    elif args.apex_amp or args.native_amp:
        print(
            "Warning: Neither APEX or native Torch AMP is available, using float32. "
            "Install NVIDA apex or upgrade to PyTorch 1.6"
        )

    # fix the seed for reproducibility
    set_seed(args.seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_train, data_loader_val = create_loaders(
        dataset_train, dataset_val, distributed=args.distributed
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = get_model(args)

    if args.flops:
        if not has_fvcore:
            print("Please install fvcore first for FLOPs calculation.")
        else:
            # Set model to evaluation mode for analysis.
            model_mode = model.training
            model.eval()
            fake_input = torch.rand(1, 3, 224, 224)
            flops_dict, *_ = flop_count(
                model, fake_input, supported_ops={"torchvision::deform_conv2d": sfc_flop_jit}
            )
            count = sum(flops_dict.values())
            model.train(model_mode)
            print("=" * 30)
            print("fvcore MAdds: {:.3f} G".format(count))

    # This part is not changed from DeiT, should be refactored for allMLP
    if args.finetune:
        model = prepare_for_finetune(model)

    model.to(device)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model)

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        print("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        print("Using native Torch AMP. Training in mixed precision.")
    else:
        print("AMP not enabled. Training in float32.")

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        if has_apex and use_amp != "native":
            # Apex DDP preferred unless native amp is activated
            model = ApexDDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    print("=" * 30)

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = create_criterion(args)

    teacher_models = None
    if args.distillation_type != "none":
        # assert args.teacher_path, 'need to specify teacher-path when using distillation'
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_models = []
        if args.data_set == "CIFAR":
            for teacher in args.teacher_model:
                teacher_model = torch.hub.load(
                    "chenyaofo/pytorch-cifar-models", f"cifar100_{teacher}", pretrained=True
                )
                teacher_models.append(teacher_model)
        elif args.data_set == "INAT":
            pass
        else:  # IMNET
            for teacher in args.teacher_model:
                teacher_model = create_model(
                    teacher,
                    pretrained=True,
                )
                teacher_models.append(teacher_model)

        for teacher_model in teacher_models:
            teacher_model.to(device)
            teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    if args.distillation_type != "none":
        criterion = DistillationLoss(
            criterion,
            teacher_models,
            args.distillation_type,
            args.distillation_alpha,
            args.distillation_tau,
            run=neptune_run,
        )

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(args.output_dir)
        / f"{args.data_set}_{args.model}_s{args.patch_size}_{args.input_size}_{current_time}"
    )
    if not args.eval:
        output_dir.mkdir(exist_ok=False, parents=True)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint_path = Path(args.resume) / "checkpoint.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

    if args.eval:
        test_stats = evaluate(
            data_loader_val, model, device, amp_autocast=amp_autocast, neptune_run=neptune_run
        )
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        return

    train(
        model,
        device=device,
        data_loader_train=data_loader_train,
        data_loader_val=data_loader_val,
        criterion=criterion,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
        mixup_fn=mixup_fn,
        amp_autocast=amp_autocast,
        lr_scheduler=lr_scheduler,
        output_dir=output_dir,
        model_without_ddp=model_without_ddp,
        dataset_val=dataset_val,
        n_parameters=n_parameters,
        args=args,
        neptune_run=neptune_run,
    )
    if neptune_run:
        neptune_run.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DeiT training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
