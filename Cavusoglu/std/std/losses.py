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
Implements the knowledge distillation loss and confidence reweighting.
This file is taken and adapted for STD implementation from Facebook
Research DeiT repository. See the original source below
https://github.com/facebookresearch/deit/blob/main/datasets.py
"""
from typing import List, Optional

import torch
from neptune import Run
from torch.nn import functional as F


def confidence_reweighting(teacher_outputs, presoftmax: bool = True):
    # Compute negative entropy
    teacher_outputs = torch.stack(teacher_outputs)
    if presoftmax:
        teacher_outputs = torch.softmax(teacher_outputs, -1)
    negative_entropy = -torch.sum(teacher_outputs * torch.log(teacher_outputs), dim=(1, 2))
    return torch.softmax(negative_entropy, -1)


class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """

    def __init__(
        self,
        base_criterion: torch.nn.Module,
        teacher_models: List[torch.nn.Module],
        distillation_type: str,
        alpha: float,
        tau: float,
        run: Optional[Run] = None,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_models = teacher_models
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        self.run = run

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == "none":
            return base_loss

        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )
        # don't backprop throught the teacher
        teacher_outputs = []
        with torch.no_grad():
            for teacher in self.teacher_models:
                teacher_outputs.append(teacher(inputs))

        teachers_weights = confidence_reweighting(teacher_outputs)
        if self.run is not None:
            for i, w in enumerate(teachers_weights):
                self.run[f"train/weight_teacher_{i}"].append(w.item())
        if self.distillation_type == "soft":
            distillation_loss = 0.0
            T = self.tau
            # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
            # with slight modifications
            for i, output_kd in enumerate(outputs_kd):
                distillation_loss += teachers_weights[i] * (
                    F.kl_div(
                        F.log_softmax(outputs_kd / T, dim=1),
                        F.log_softmax(teacher_outputs[i] / T, dim=1),
                        reduction="sum",
                        log_target=True,
                    )
                    * (T * T)
                    / outputs_kd.numel()
                )
        else:  # hard distillation
            distillation_loss = 0.0
            for i, output_kd in enumerate(outputs_kd):
                distillation_loss += teachers_weights[i] * F.cross_entropy(
                    output_kd, teacher_outputs[i].argmax(dim=1)
                )

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
