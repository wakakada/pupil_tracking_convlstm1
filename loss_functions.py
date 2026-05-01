import torch
import torch.nn as nn
import torch.nn.functional as F

class EuclideanDistanceLoss(nn.Module):
    """
    欧氏距离损失函数，专门用于坐标回归任务
    直接优化预测坐标与真实坐标之间的欧氏距离
    """
    def __init__(self, weight=1.0):
        super(EuclideanDistanceLoss, self).__init__()
        self.weight = weight

    def forward(self, pred_coords, gt_coords):
        """
        pred_coords: 预测的坐标 (batch_size, 2) - 归一化的坐标
        gt_coords: 真实的坐标 (batch_size, 2) - 归一化的坐标
        """
        # 计算欧氏距离
        euclidean_dist = torch.sqrt(torch.sum((pred_coords - gt_coords) ** 2, dim=1))
        # 返回平均距离作为损失
        loss = torch.mean(euclidean_dist)
        return loss * self.weight, {'euclidean_loss': loss.item()}