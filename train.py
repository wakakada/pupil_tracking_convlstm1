import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
from video_dataset import VideoSequenceDataset
from model import PupilTrackingConvLSTM
from loss_functions import CombinedLoss
import os
from tqdm import tqdm
import numpy as np
import gc

def create_datasets():
    """创建训练和验证数据集，只调用一次以避免重复加载"""
    # 获取所有受试者编号
    all_subjects = [int(d) for d in os.listdir(LPW_ROOT) if os.path.isdir(os.path.join(LPW_ROOT, d)) and d.isdigit()]
    all_subjects.sort()

    # 随机种子
    if RANDOM_SEED:
        random.seed(RANDOM_SEED)
    random.shuffle(all_subjects)

    # 按比例划分
    split_idx = int(len(all_subjects) * TRAIN_RATIO)
    train_subjects = all_subjects[:split_idx]
    val_subjects = all_subjects[split_idx:]

    print(f"数据集划分（按受试者，共 {len(all_subjects)} 个）:")
    print(f"Train subjects:{train_subjects}")
    print(f"Val subjects:{val_subjects}")

    # 创建训练和验证数据集（使用序列数据集）
    train_dataset = VideoSequenceDataset(
        LPW_ROOT, train_subjects, 
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        sequence_length=SEQUENCE_LENGTH,  # 序列长度
        augment_prob=AUGMENT_PROBABILITY
    )
    val_dataset = VideoSequenceDataset(
        LPW_ROOT, val_subjects, 
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        sequence_length=SEQUENCE_LENGTH,  # 序列长度
        augment_prob=0.0  # 验证集不使用数据增强
    )
    
    return train_dataset, val_dataset

def main():
    train_dataset, val_dataset = create_datasets()
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        raise ValueError(f"训练数据集为空，请检查训练受试者数据是否存在")
    if len(val_dataset) == 0:
        raise ValueError(f"验证数据集为空，请检查验证受试者数据是否存在")
    
    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"验证数据集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 使用改进的ConvLSTM模型（包含分割和回归分支）
    model = PupilTrackingConvLSTM(
        input_dim=1,           # 灰度图像
        hidden_dim=64,         # 隐藏层维度
        kernel_size=(3, 3),    # 卷积核大小
        num_layers=2,          # ConvLSTM层数
        dropout_rate=0.5       # Dropout率
    ).to(DEVICE)
    
    # 使用组合损失函数
    criterion = CombinedLoss(seg_weight=0.8, reg_weight=1.0, smooth_l1_weight=0.3)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3, eps=1e-8)
    # 使用余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # 早停机制相关参数
    best_val_loss = float('inf')
    patience = EARLY_STOPPING_PATIENCE  # 允许验证损失不下降的最大epoch数
    patience_counter = 0  # 计数器
    min_delta = EARLY_STOPPING_MIN_DELTA # 认为是改善的最小变化量

    for epoch in range(1, EPOCHS+1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        # 使用tqdm包装训练数据加载器，显示训练进度
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training", leave=False)
        for data, targets in train_pbar:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            # 改进的ConvLSTM模型，需要返回分割掩码和回归输出
            preds, seg_masks, reg_outputs = model(data, return_segmentation=True)
            
            # 使用组合损失函数
            loss, loss_components = criterion(
                pred_coords=preds,
                gt_coords=targets,
                pred_seg=seg_masks,
                gt_seg=None,  # 如果有真实分割标签可以在这里提供
                pred_reg=reg_outputs
            )
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
            # 更新进度条显示当前损失
            train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        train_loss /= len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        # 随机选择一个验证样本用于展示预测结果
        display_sample = True
        
        # 使用tqdm包装验证数据加载器，显示验证进度
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Validation", leave=False)
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_pbar):
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                
                # 改进的ConvLSTM模型
                preds, seg_masks, reg_outputs = model(data, return_segmentation=True)
                
                # 使用组合损失函数计算验证损失
                loss, _ = criterion(
                    pred_coords=preds,
                    gt_coords=targets,
                    pred_seg=seg_masks,
                    gt_seg=None,  # 如果有真实分割标签可以在这里提供
                    pred_reg=reg_outputs
                )
                
                val_loss += loss.item() * data.size(0)
                
                # 随机选择一个验证样本展示预测结果
                if display_sample and batch_idx == 0:  # 在第一个批次中选择一个样本展示
                    # 随机选择批次中的一个样本
                    sample_idx = random.randint(0, data.size(0)-1)
                    
                    sample_data = data[sample_idx:sample_idx+1]  # 保持batch维度
                    sample_target = targets[sample_idx]
                    sample_pred, _, _ = model(sample_data, return_segmentation=True)
                    
                    # 将归一化坐标转换回像素坐标（原始图像尺寸为640x480）
                    orig_w, orig_h = 640, 480  # 原始图像尺寸
                    pred_x_pixel = sample_pred[0, 0].item() * orig_w
                    pred_y_pixel = sample_pred[0, 1].item() * orig_h
                    target_x_pixel = sample_target[0].item() * orig_w
                    target_y_pixel = sample_target[1].item() * orig_h
                    
                    print(f"\n随机验证样本预测结果 (Epoch {epoch}):")
                    print(f"  预测坐标: ({pred_x_pixel:.2f}, {pred_y_pixel:.2f})")
                    print(f"  真实坐标: ({target_x_pixel:.2f}, {target_y_pixel:.2f})")
                    print(f"  坐标误差: {np.sqrt((pred_x_pixel - target_x_pixel)**2 + (pred_y_pixel - target_y_pixel)**2):.2f} 像素")
                    
                    display_sample = False  # 确保只显示一次
                
                # 更新进度条显示当前损失
                val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

        val_loss /= len(val_loader.dataset)
        scheduler.step()    # 调整学习率

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # 早停机制
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, MODEL_SAVE_PATH)
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            print(f"  -> No improvement in validation loss for {patience_counter}/{patience} epochs")
        
        # 检查是否需要早停
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            print(f"Best validation loss was {best_val_loss:.6f}")
            break
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\nTraining completed. Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()