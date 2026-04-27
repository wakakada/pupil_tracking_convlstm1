# loss_functions.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=1.0, reg_weight=1.0, smooth_l1_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.reg_weight = reg_weight
        self.smooth_l1_weight = smooth_l1_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()

    def forward(self, pred_coords, gt_coords, pred_seg=None, gt_seg=None, pred_reg=None):
        # 初始化损失变量（避免 UnboundLocalError）
        device = pred_coords.device
        seg_loss = torch.tensor(0.0, device=device)
        direct_reg_loss = torch.tensor(0.0, device=device)

        # 回归损失（坐标预测）
        reg_loss = self.mse_loss(pred_coords, gt_coords)
        total_loss = self.reg_weight * reg_loss

        # 分割损失（如有提供真实分割标签）
        if pred_seg is not None and gt_seg is not None:
            seg_loss = self.bce_loss(pred_seg, gt_seg)
            total_loss += self.seg_weight * seg_loss

        # 直接回归损失（如从分割分支得到的额外回归输出）
        if pred_reg is not None:
            direct_reg_loss = self.smooth_l1_loss(pred_reg, gt_coords)
            total_loss += self.smooth_l1_weight * direct_reg_loss

        # 返回总损失及各分量（所有损失均为标量）
        loss_dict = {
            'reg_loss': reg_loss.item(),
            'seg_loss': seg_loss.item(),
            'direct_reg_loss': direct_reg_loss.item()
        }
        return total_loss, loss_dict


class BoundaryAwareLoss(nn.Module):
    def __init__(self, seg_weight=1.0, reg_weight=1.0, boundary_weight=0.5):
        super(BoundaryAwareLoss, self).__init__()
        self.seg_weight = seg_weight
        self.reg_weight = reg_weight
        self.boundary_weight = boundary_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred_coords, gt_coords, pred_seg=None, gt_seg=None, pred_reg=None):
        total_loss = 0
        
        # 坐标回归损失
        reg_loss = self.mse_loss(pred_coords, gt_coords)
        total_loss += self.reg_weight * reg_loss
        
        # 如果提供了分割预测和标签，则计算分割损失
        if pred_seg is not None and gt_seg is not None:
            seg_loss = self.bce_loss(pred_seg, gt_seg)
            total_loss += self.seg_weight * seg_loss
            
            # 边界感知损失
            if self.boundary_weight > 0:
                boundary_loss = self.boundary_loss(pred_seg, gt_seg)
                total_loss += self.boundary_weight * boundary_loss
        
        # 如果提供了直接回归输出，则也计算其损失
        if pred_reg is not None:
            direct_reg_loss = self.mse_loss(pred_reg, gt_coords)
            total_loss += self.reg_weight * direct_reg_loss
        
        return total_loss, {'reg_loss': reg_loss.item(), 'seg_loss': seg_loss.item() if pred_seg is not None else 0, 'boundary_loss': boundary_loss.item() if pred_seg is not None and gt_seg is not None and self.boundary_weight > 0 else 0}

    def boundary_loss(self, pred, target):
        """计算边界感知损失"""
        # 计算边缘
        edge_pred = self.compute_edge_map(pred)
        edge_target = self.compute_edge_map(target)
        
        # 边界上的MSE损失
        return F.mse_loss(edge_pred, edge_target)

    def compute_edge_map(self, x):
        """计算边缘图"""
        # 使用Sobel算子计算边缘
        sobel_x = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32, device=x.device)
        sobel_y = torch.tensor([ [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] ], dtype=torch.float32, device=x.device)
        
        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        
        # 计算梯度幅值
        edge_map = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return edge_map