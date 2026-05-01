import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
from video_dataset import VideoSequenceDataset
from model import PupilTrackingConvLSTM
from loss_functions import EuclideanDistanceLoss
from sklearn.model_selection import KFold
import os
from tqdm import tqdm
import numpy as np
import gc
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device):
    """评估模型性能，返回各种指标"""
    model.eval()
    criterion = EuclideanDistanceLoss(weight=1.0)  # 直接使用欧氏距离损失函数评估坐标误差
    total_loss = 0.0
    total_distance_error = 0.0
    total_samples = 0
    all_errors = []

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)
            
            # 模型预测，返回坐标和分割输出
            center_coords, final_mask, regression_output = model(data, return_segmentation=True)
            preds = center_coords  # 使用融合后的坐标作为主要预测
            seg_output = final_mask  # 分割输出
            
            # 计算损失，只使用坐标回归部分
            loss, _ = criterion(
                pred_coords=preds,
                gt_coords=targets
            )
            
            total_loss += loss.item() * data.size(0)
            
            # 计算欧氏距离误差 (像素级别)
            orig_w, orig_h = 640, 480
            pred_pixels = preds.cpu() * torch.tensor([orig_w, orig_h], dtype=torch.float32)
            target_pixels = targets.cpu() * torch.tensor([orig_w, orig_h], dtype=torch.float32)
            
            # 计算欧氏距离
            distances = torch.sqrt(torch.sum((pred_pixels - target_pixels)**2, dim=1))
            total_distance_error += torch.sum(distances).item()
            all_errors.extend(distances.numpy())
            
            total_samples += data.size(0)

    avg_loss = total_loss / total_samples
    avg_distance_error = total_distance_error / total_samples
    median_distance_error = np.median(all_errors)
    rmse = np.sqrt(np.mean(np.array(all_errors) ** 2))
    std_error = np.std(all_errors)

    return {
        'avg_loss': avg_loss,
        'avg_distance_error': avg_distance_error,
        'median_distance_error': median_distance_error,
        'rmse': rmse,
        'std_error': std_error
    }

def create_fold_datasets(all_subjects, train_indices, val_indices):
    """根据索引创建特定折的数据集"""
    train_subjects = [all_subjects[i] for i in train_indices]
    val_subjects = [all_subjects[i] for i in val_indices]
    
    print(f"Fold - Train subjects: {train_subjects}")
    print(f"Fold - Val subjects: {val_subjects}")

    # 创建训练、验证数据集（使用序列数据集）
    train_dataset = VideoSequenceDataset(
        LPW_ROOT, train_subjects, 
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        sequence_length=SEQUENCE_LENGTH,  # 序列长度
        augment_prob=AUGMENT_PROBABILITY,
        data_name="Train"
    )
    val_dataset = VideoSequenceDataset(
        LPW_ROOT, val_subjects, 
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        sequence_length=SEQUENCE_LENGTH,  # 序列长度
        augment_prob=0.0,  # 验证集不使用数据增强
        data_name="Val"
    )
    
    return train_dataset, val_dataset

def validate_model(epoch, model, val_loader, criterion, device, fold):
    """验证模型并返回验证损失及显示样本预测结果"""
    model.eval()
    val_loss = 0.0
    display_sample = True
    
    # 使用tqdm包装验证数据加载器，显示验证进度
    val_pbar = tqdm(val_loader, desc=f"Fold {fold+1}/{k_folds} - Epoch {epoch}/{EPOCHS} - Validation", leave=True)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_pbar):
            data, targets = data.to(device), targets.to(device)
            
            center_coords, final_mask, regression_output = model(data, return_segmentation=True)
            preds = center_coords  # 使用融合后的坐标作为主要预测
            seg_output = final_mask  # 分割输出
            
            # 使用组合损失函数计算验证损失
            loss, _ = criterion(
                pred_coords=preds,
                gt_coords=targets
            )
            
            val_loss += loss.item() * data.size(0)
            
            # 随机选择一个验证样本展示预测结果
            if display_sample and batch_idx == 0:  # 在第一个批次中选择一个样本展示
                # 随机选择批次中的一个样本
                sample_idx = random.randint(0, data.size(0)-1)
                
                sample_data = data[sample_idx:sample_idx+1]  # 保持batch维度
                sample_target = targets[sample_idx]
                sample_center_coords, sample_final_mask, sample_regression_output = model(sample_data, return_segmentation=True)
                sample_pred = sample_center_coords  # 使用融合后的坐标作为主要预测
                
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
    return val_loss


def main():
    # 获取所有受试者编号
    all_subjects = [int(d) for d in os.listdir(LPW_ROOT) if os.path.isdir(os.path.join(LPW_ROOT, d)) and d.isdigit()]
    all_subjects.sort()

    # 设置随机种子
    if RANDOM_SEED:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)
    
    # 固定一部分数据作为测试集，其余部分进行K折交叉验证
    n_total_subjects = len(all_subjects)
    n_test_subjects = max(1, int(n_total_subjects * 0.2))  # 使用20%的数据作为固定测试集
    n_cv_subjects = n_total_subjects - n_test_subjects     # 剩余80%用于交叉验证
    
    # 分离测试集和用于交叉验证的数据集
    test_subjects = all_subjects[:n_test_subjects]
    cv_subjects = all_subjects[n_test_subjects:]
    
    # 创建固定的测试数据集
    test_dataset = VideoSequenceDataset(
        LPW_ROOT, test_subjects, 
        img_size=(IMG_HEIGHT, IMG_WIDTH), 
        sequence_length=SEQUENCE_LENGTH,
        augment_prob=0.0,  # 测试集不使用数据增强
        data_name="Test"
    )
    
    print(f"固定测试集受试者: {test_subjects}")
    print(f"交叉验证受试者: {cv_subjects}")

    # 计算交叉验证中每折的大小
    fold_size = n_cv_subjects // k_folds
    fold_results = []
    
    # 执行K折交叉验证，但只在cv_subjects上进行
    for fold in range(k_folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{k_folds}")
        print(f"{'='*50}")
        
        # 确定验证集的索引（在cv_subjects中）
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else len(cv_subjects)
        val_indices = list(range(val_start, val_end))
        
        # 其余的作为训练集（在cv_subjects中）
        train_indices = list(range(0, val_start)) + list(range(val_end, len(cv_subjects)))
        
        # 映射回原始的subject ID
        train_subjects = [cv_subjects[i] for i in train_indices]
        val_subjects = [cv_subjects[i] for i in val_indices]
        
        print(f"Fold - Train subjects: {train_subjects}")
        print(f"Fold - Val subjects: {val_subjects}")

        # 创建训练、验证数据集（使用序列数据集）
        train_dataset = VideoSequenceDataset(
            LPW_ROOT, train_subjects, 
            img_size=(IMG_HEIGHT, IMG_WIDTH), 
            sequence_length=SEQUENCE_LENGTH,  # 序列长度
            augment_prob=AUGMENT_PROBABILITY,
            data_name="Train"
        )
        val_dataset = VideoSequenceDataset(
            LPW_ROOT, val_subjects, 
            img_size=(IMG_HEIGHT, IMG_WIDTH), 
            sequence_length=SEQUENCE_LENGTH,  # 序列长度
            augment_prob=0.0,  # 验证集不使用数据增强
            data_name="Val"
        )

        # 检查数据集是否为空
        if len(train_dataset) == 0:
            raise ValueError(f"训练数据集为空，请检查训练受试者数据是否存在")
        if len(val_dataset) == 0:
            raise ValueError(f"验证数据集为空，请检查验证受试者数据是否存在")
        if len(test_dataset) == 0:
            raise ValueError(f"测试数据集为空，请检查测试受试者数据是否存在")
        
        print(f"训练数据集大小: {len(train_dataset)}")
        print(f"验证数据集大小: {len(val_dataset)}")
        print(f"测试数据集大小: {len(test_dataset)}")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        # 使用改进的ConvLSTM模型（包含回归分支）
        model = PupilTrackingConvLSTM(
            input_dim=1,           # 灰度图像
            hidden_dim=64,         # 隐藏层维度
            kernel_size=(3, 3),    # 卷积核大小
            num_layers=2,          # ConvLSTM层数
            dropout_rate=0.6       # Dropout率
        ).to(DEVICE)
        
        criterion = EuclideanDistanceLoss(weight=1.0)  # 直接使用欧氏距离损失函数优化坐标预测
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3, eps=1e-8)
        # 使用余弦退火调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        # 早停机制相关参数
        best_val_loss = float('inf')
        patience = EARLY_STOPPING_PATIENCE  # 允许验证损失不下降的最大epoch数
        patience_counter = 0  # 计数器
        min_delta = EARLY_STOPPING_MIN_DELTA # 认为是改善的最小变化量

        train_losses = []
        val_losses = []

        for epoch in range(1, EPOCHS+1):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            # 使用tqdm包装训练数据加载器，显示训练进度
            train_pbar = tqdm(train_loader, desc=f"Fold {fold+1}/{k_folds} - Epoch {epoch}/{EPOCHS} - Training ", leave=True)
            for data, targets in train_pbar:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                
                # 改进的ConvLSTM模型，返回回归输出
                center_coords, final_mask, regression_output = model(data, return_segmentation=True)
                preds = center_coords  # 使用融合后的坐标作为主要预测
                seg_output = final_mask  # 分割输出
                
                # 使用欧氏距离损失函数
                loss, _ = criterion(
                    pred_coords=preds,
                    gt_coords=targets
                )
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)
                
                # 更新进度条显示当前损失
                train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

            train_loss /= len(train_loader.dataset)

            # 验证阶段
            val_loss = validate_model(epoch, model, val_loader, criterion, DEVICE, fold)
            scheduler.step()    # 调整学习率

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Fold {fold+1} - Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # 早停机制
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                # 为每个fold保存不同的模型
                fold_model_path = f"{MODEL_SAVE_PATH.split('.')[0]}_fold_{fold+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, fold_model_path)
                print(f"  -> Saved best model for fold {fold+1} (val_loss={val_loss:.6f})\n")
                patience_counter = 0  # 重置计数器
            else:
                patience_counter += 1
                print(f"  -> No improvement in validation loss for {patience_counter}/{patience} epochs\n")
            
            # 检查是否需要早停
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch} for fold {fold+1}!")
                print(f"Best validation loss was {best_val_loss:.6f}")
                break
            
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        print(f"\nFold {fold+1} completed. Best validation loss: {best_val_loss:.6f}")

        # 绘制并显示loss曲线
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs_range, val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.scatter(epochs_range, train_losses, c='blue', s=30, zorder=5)
        plt.scatter(epochs_range, val_losses, c='red', s=30, zorder=5)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Fold {fold+1}/{k_folds} - Training & Validation Loss', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_path = f"loss_curve_fold_{fold+1}.png"
        plt.savefig(save_path, dpi=150)
        print(f"Loss curve saved to: {save_path}")
        plt.show()

        # 加载最佳模型进行测试（在固定的测试集上）
        print(f"\nLoading best model for fold {fold+1} testing...")
        fold_model_path = f"{MODEL_SAVE_PATH.split('.')[0]}_fold_{fold+1}.pth"
        checkpoint = torch.load(fold_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 测试模型性能（在固定的测试集上）
        print(f"Evaluating model on fixed test set...")
        test_metrics = evaluate_model(model, test_loader, DEVICE)
        
        print(f"\nFold {fold+1} 固定测试集性能指标:")
        print(f"  平均损失: {test_metrics['avg_loss']:.6f}")
        print(f"  平均距离误差: {test_metrics['avg_distance_error']:.2f} 像素")
        print(f"  中位数距离误差: {test_metrics['median_distance_error']:.2f} 像素")
        print(f"  RMSE: {test_metrics['rmse']:.2f} 像素")
        print(f"  误差标准差: {test_metrics['std_error']:.2f} 像素")
        
        fold_results.append(test_metrics)
    
    # 计算并输出交叉验证的平均结果
    print(f"\n{'='*50}")
    print("CROSS VALIDATION RESULTS SUMMARY")
    print(f"{'='*50}")
    
    avg_metrics = {}
    metric_names = ['avg_loss', 'avg_distance_error', 'median_distance_error', 'rmse', 'std_error']
    
    for metric in metric_names:
        values = [result[metric] for result in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        avg_metrics[metric] = {'mean': mean_val, 'std': std_val}
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print(f"\n交叉验证完成! 使用了 {k_folds} 折交叉验证")
    print(f"测试集占总数据的约 {len(test_subjects)/len(all_subjects)*100:.1f}%")


if __name__ == "__main__":
    main()