from torch import nn

class SimilarityLoss(nn.Module):
    def forward(self, p, z):
        z = z.detach()  # stop gradient
        return - nn.functional.cosine_similarity(p, z, dim=-1).mean()
