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


class HeatmapLoss(nn.Module):
    """热力图监督损失：预测热力图对齐到 GT 坐标生成的高斯热斑"""
    def __init__(self, h, w, sigma=3.0):
        super().__init__()
        self.h, self.w = h, w
        self.sigma = sigma
        self.mse = nn.MSELoss()

    def forward(self, heatmap, gt_coords):
        """
        heatmap: (B, 1, H, W)  sigmoid 输出，值域 [0,1]
        gt_coords: (B, 2)  归一化坐标 [x, y]
        """
        B = heatmap.size(0)
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32, device=heatmap.device),
            torch.arange(self.w, dtype=torch.float32, device=heatmap.device),
            indexing='ij'
        )
        cx = gt_coords[:, 0] * self.w  # (B,)
        cy = gt_coords[:, 1] * self.h  # (B,)
        dist_sq = (xx - cx.view(B, 1, 1)) ** 2 + (yy - cy.view(B, 1, 1)) ** 2
        target = torch.exp(-dist_sq / (2 * self.sigma ** 2)).unsqueeze(1)  # (B, 1, H, W)
        return self.mse(heatmap, target)