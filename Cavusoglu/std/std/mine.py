# MIT License
# Copyright (c) 2023 Devrim Cavusoglu
"""
Implementation of MINE objective [1] and MINE regularization of the
STD framework.

[1] https://arxiv.org/abs/1801.04062
"""
from typing import Optional

import torch
from neptune import Run
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from torch.optim import SGD, Optimizer


class MINEObjective(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super(MINEObjective, self).__init__(size_average, reduce, reduction)

    def forward(
        self, joint: Tensor, marginal: Tensor, cached_avg: float = None, momentum: float = 0.99
    ) -> Tensor:
        joint_avg = torch.mean(joint)
        marginal_avg = torch.mean(torch.exp(marginal))
        mine = joint_avg - torch.log(marginal_avg)

        # EMA & bias correction
        # cached_avg = (1 - momentum) * marginal_avg + momentum * cached_avg
        return mine


def build_mine(model: nn.Module, dim_spatial: int, dim_channel: int, device: torch.device):
    mine_network = nn.Sequential(
        nn.Linear(dim_spatial + dim_channel, 512),
        nn.GELU(),
        nn.Linear(512, 512),
        nn.GELU(),
        nn.Linear(512, 1),
    )
    mine_network.to(device)
    model_optimizer = SGD(params=model.parameters(), lr=0.01, momentum=0)
    mine_optimizer = SGD(params=mine_network.parameters(), lr=0.01, momentum=0, maximize=True)
    objective = MINEObjective()
    return model_optimizer, mine_network, mine_optimizer, objective


def mine_regularization(
    model: nn.Module,
    mine_network: nn.Module,
    model_optimizer: Optimizer,
    mine_optimizer: Optimizer,
    objective: MINEObjective,
    x: Tensor,
    neptune_run: Optional[Run] = None,
):
    """
    MINE algorithm for regularizing the distilled spatial-channel tokens to disentangle from
    `Algorithm 1`.

    Notes:
            Index permutation step is set as in the simplest form where the indices are permuted
            by shifting to right (j = i + 1). This simple idea guarantees derangement (j != i).
            However, the derangement might be refactored with better permutations.

    Args:
            model: STD vision model having distillation tokens.
            mine_network: Statistics network that MINE uses in regularization.
            model_optimizer: MINE optimizer for model being regularized for MINE algorithm, this is
                    not the optimizer that trains the vision `model`. This performs gradient descent on
                    `model`.
            mine_optimizer: MINE optimizer for statistics network (i.e. `mine_network`). This
                    performs gradient ascent on `mine_network`.
            objective: MINE objective for computing the neural information measure.
            x: Samples for MINE algorithm to be used.
    """
    print("Optimizing MINE..")
    model.train()
    mine_network.train()
    for p in model.parameters():
        model_device = p.device
    x = x.to(model_device, non_blocking=True)
    B = x.shape[0]
    _, ts, tc = model.forward_features(x)

    # create indicies for marginal dist (unpaired tokens)
    idx = torch.arange(B) + 1  # shift index to right
    idx[-1] = 0  # last-pos correction

    # Avg. across teacher dist heads
    ts, tc = ts.mean(dim=1), tc.mean(dim=2)  # Shape (N,L)
    joint = torch.hstack((ts, tc))  # paired tokens
    marginal = torch.hstack((ts, tc[idx]))  # unpaired tokens (j = i + 1)

    joint_outputs = mine_network(joint)
    marginal_outputs = mine_network(marginal)

    # Weight updates
    mine_optimizer.zero_grad()
    model_optimizer.zero_grad()
    J = objective(joint_outputs, marginal_outputs)
    print(f"MINE Objective: {J:.5f}")
    if neptune_run:
        neptune_run["train/mine_objective"].append(J.item())
    J.backward()
    mine_optimizer.step()  # apply gradient ascent
    model_optimizer.step()  # apply gradient descent
