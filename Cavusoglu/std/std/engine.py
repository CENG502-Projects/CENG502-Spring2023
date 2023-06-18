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
Train and eval functions used in main.py. This file is taken and adapted
for STD implementation from Facebook Research DeiT repository. See the
original source below
https://github.com/facebookresearch/deit/blob/main/engine.py
"""
import math
import sys
import warnings
from typing import Iterable, Optional

import numpy as np
import torch
from neptune import Run
from timm.data import Mixup
from timm.utils import ModelEma, accuracy

import std.utils as utils
from std.losses import DistillationLoss


def train_one_epoch(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    distillation_type: str,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode=True,
    amp_autocast=None,
    n_mine_samples: int = None,
    neptune_run: Optional[Run] = None,
):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ", neptune_run=neptune_run)
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if n_mine_samples == 1:
        # We cannot pick joint vs marginal with sample size = 1.
        warnings.warn(
            "Given `n_mine_samples` must be greater than 1 (if not 0), "
            "changing the number to be set as default (batch-size)"
        )
        n_mine_samples = None
    if distillation_type == "none":
        n_mine_samples = 0

    n_mine_samples = data_loader.batch_size if n_mine_samples is None else n_mine_samples
    mine_samples = []
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if n_mine_samples > 1 and len(mine_samples) < n_mine_samples:
            sample_size = int(np.ceil(data_loader.batch_size / len(data_loader)))
            mine_rnd = np.random.randint(data_loader.batch_size, size=sample_size)
            mine_sample = samples[mine_rnd].expand(len(mine_rnd), -1, -1, -1)
            mine_samples.append(mine_sample)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with amp_autocast():
            outputs = model(samples)
            if distillation_type == "none":
                loss = criterion(outputs, targets)
            else:
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )
        else:
            loss.backward(create_graph=is_second_order)
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if len(mine_samples) > 0:
        mine_samples = torch.cat(mine_samples, 0)
        return stats, mine_samples
    else:
        return stats, None


@torch.no_grad()
def evaluate(data_loader, model, device, amp_autocast=None, neptune_run: Optional = None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ", neptune_run=neptune_run, is_train=False)
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with amp_autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    metric_logger.log_neptune(key="acc@1", val=metric_logger.acc1.global_avg)
    metric_logger.log_neptune(key="acc@5", val=metric_logger.acc5.global_avg)
    metric_logger.log_neptune(key="loss", val=metric_logger.loss.global_avg)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
