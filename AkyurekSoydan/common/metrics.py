""" Code below is used for mIoU and FB-mIoU evaluation metrics.
    
    DISCLAIMER: It is based on https://github.com/juhongm999/hsnet/tree/main

"""
import torch

class Metrics():
    
    def __init__(self,  device, num_class, dataset):
        
        self.class_ids_interest = torch.tensor(dataset.class_ids).to(device)
        
        self.nclass = num_class
        self.intersection_buf = torch.zeros([2, self.nclass]).to(device)
        self.union_buf = torch.zeros([2, self.nclass]).to(device)
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []
        self.device = device

    def update(self, inter_b, union_b, class_id, loss):
        self.intersection_buf.index_add_(1, class_id, inter_b.to(self.device))
        self.union_buf.index_add_(1, class_id, union_b.to(self.device))
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_iou(self):
        iou = self.intersection_buf.float() / torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean() * 100

        return miou, fb_iou