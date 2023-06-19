# Author: Utku Mert Topçuoğlu + https://github.com/Megvii-BaseDetection/OTA/blob/2c85b4d0f9031396854aae969330dde2ab5eacbd/playground/detection/coco/ota.x101.fpn.coco.800size.1x/fcos.py#L665

# Discussion about the difference between this and the original sinkhorn algorithms implementation. This one is applying something like momentum: https://chat.openai.com/share/710bf589-eb7f-4c62-87d9-8a587c7b1105
# I took it from https://github.com/Megvii-BaseDetection/OTA/blob/2c85b4d0f9031396854aae969330dde2ab5eacbd/playground/detection/coco/ota.x101.fpn.coco.800size.1x/fcos.py#L665
# and modified accordingly (u and v update order were confused).
# OTA took it from https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py and modified without mentioning.
import torch

class SinkhornDistance(torch.nn.Module):
    r"""
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
    """

    def __init__(self, eps=1e-1, max_iter=50): # Default values from the paper OTA (and UniVIP)
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, mu, nu, C):
        # NOTE checked u-a-demander v-b-supplier correspondance from OTA paper.
        # mu is demander a -> (batch_size, instance numbers K, 1), nu is supplier b -> (batch_size, instance numbers K, 1) -> you have to squeeze them.
        mu, nu = mu.squeeze(dim=-1), nu.squeeze(dim=-1) 
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        # C (batch_size, instance numbers K, instance numbers K)

        # Sinkhorn iterations
        for _ in range(self.max_iter):
            # NOTE original algorithm first updates u then v. Hence modified. UniVIP Li_i is kind of symmetric though.
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            # transpose((-2, -1)) to avoid batch dimension
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V)).detach()
        # Sinkhorn distance
        # transpose((-2, -1)) to avoid batch dimension
        cost = torch.sum(pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        # Original code updates (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon, but here we avoid batch dimension.
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
