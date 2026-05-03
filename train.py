import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
from video_dataset import VideoSequenceDataset, build_video_items
from model import PupilTrackingConvLSTM
from loss_functions import EuclideanDistanceLoss, HeatmapLoss
from tqdm import tqdm
import numpy as np
import gc
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, heatmap_criterion):
    """评估模型性能，返回各种指标"""
    model.eval()
    criterion = EuclideanDistanceLoss(weight=1.0)
    total_loss = 0.0
    total_distance_error = 0.0
    total_samples = 0
    all_errors = []

    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)

            preds, seg_output = model(data, return_segmentation=True)

            coord_loss, _ = criterion(pred_coords=preds, gt_coords=targets)
            hm_loss = heatmap_criterion(seg_output, targets)
            loss = coord_loss + HEATMAP_LOSS_WEIGHT * hm_loss

            total_loss += loss.item() * data.size(0)

            # 计算欧氏距离误差 (像素级别)
            orig_w, orig_h = 640, 480
            pred_pixels = preds.cpu() * torch.tensor([orig_w, orig_h], dtype=torch.float32)
            target_pixels = targets.cpu() * torch.tensor([orig_w, orig_h], dtype=torch.float32)

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


def validate_model(epoch, model, val_loader, criterion, device, fold, heatmap_criterion):
    """验证模型并返回验证损失及显示样本预测结果"""
    model.eval()
    val_loss = 0.0
    # 随机选一个 batch 来展示样本（避免总是展示同一个 batch）
    total_batches = len(val_loader)
    display_batch_idx = random.randint(0, total_batches - 1) if total_batches > 0 else 0
    display_sample = True

    val_pbar = tqdm(val_loader, desc=f"Fold {fold+1}/{k_folds} - Epoch {epoch}/{EPOCHS} - Validation", leave=True)
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(val_pbar):
            data, targets = data.to(device), targets.to(device)

            preds, seg_output = model(data, return_segmentation=True)

            coord_loss, _ = criterion(pred_coords=preds, gt_coords=targets)
            hm_loss = heatmap_criterion(seg_output, targets)
            loss = coord_loss + HEATMAP_LOSS_WEIGHT * hm_loss

            val_loss += loss.item() * data.size(0)

            if display_sample and batch_idx == display_batch_idx:
                sample_idx = random.randint(0, data.size(0)-1)

                sample_data = data[sample_idx:sample_idx+1]
                sample_target = targets[sample_idx]
                sample_pred, _ = model(sample_data, return_segmentation=True)

                orig_w, orig_h = 640, 480
                pred_x_pixel = sample_pred[0, 0].item() * orig_w
                pred_y_pixel = sample_pred[0, 1].item() * orig_h
                target_x_pixel = sample_target[0].item() * orig_w
                target_y_pixel = sample_target[1].item() * orig_h

                print(f"\n随机验证样本预测结果 (Epoch {epoch}, batch {display_batch_idx}/{total_batches}):")
                print(f"  预测坐标: ({pred_x_pixel:.2f}, {pred_y_pixel:.2f})")
                print(f"  真实坐标: ({target_x_pixel:.2f}, {target_y_pixel:.2f})")
                print(f"  坐标误差: {np.sqrt((pred_x_pixel - target_x_pixel)**2 + (pred_y_pixel - target_y_pixel)**2):.2f} 像素")

                display_sample = False

            val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

    val_loss /= len(val_loader.dataset)
    return val_loss


def main():
    # 扫描所有视频，构建视频项列表，每项为 (video_path, label_path, video_name)
    # video_name 格式: "受试者/video.avi"
    all_video_items = build_video_items(LPW_ROOT)

    if len(all_video_items) == 0:
        raise ValueError(f"未在 {LPW_ROOT} 中找到任何视频数据，请检查数据集路径")

    print(f"共发现 {len(all_video_items)} 个视频")
    for _, _, name in all_video_items[:5]:
        print(f"  示例: {name}")
    if len(all_video_items) > 5:
        print(f"  ... 等共 {len(all_video_items)} 个视频")

    # 设置随机种子
    if RANDOM_SEED:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        torch.manual_seed(RANDOM_SEED)

    # 打乱视频顺序，避免同一受试者的视频集中在前/后半段
    random.shuffle(all_video_items)

    # 固定一部分视频作为测试集，其余进行K折交叉验证
    n_total_videos = len(all_video_items)
    n_test_videos = max(1, int(n_total_videos * 0.2))  # 使用20%的视频作为固定测试集
    n_cv_videos = n_total_videos - n_test_videos       # 剩余80%用于交叉验证

    # 分离测试集和用于交叉验证的数据集
    test_video_items = all_video_items[:n_test_videos]
    cv_video_items = all_video_items[n_test_videos:]

    print(f"固定测试集视频 ({len(test_video_items)} 个):")
    for _, _, name in test_video_items:
        print(f"  {name}")
    print(f"交叉验证视频 ({len(cv_video_items)} 个)")

    # 创建固定的测试数据集
    test_dataset = VideoSequenceDataset(
        test_video_items,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        sequence_length=SEQUENCE_LENGTH,
        stride=3,
        augment_prob=0.0,  # 测试集不使用数据增强
        data_name="Test"
    )

    # 计算交叉验证中每折的大小（按视频数划分）
    fold_size = n_cv_videos // k_folds
    fold_results = []

    # 执行K折交叉验证
    for fold in range(k_folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{k_folds}")
        print(f"{'='*50}")

        # 确定验证集的索引（在cv_video_items中）
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else len(cv_video_items)

        # 按视频划分训练集和验证集
        val_video_items = cv_video_items[val_start:val_end]
        train_video_items = cv_video_items[:val_start] + cv_video_items[val_end:]

        print(f"Fold - Train videos: {len(train_video_items)}")
        print(f"Fold - Val videos: {len(val_video_items)}")

        # 创建训练、验证数据集
        train_dataset = VideoSequenceDataset(
            train_video_items,
            img_size=(IMG_HEIGHT, IMG_WIDTH),
            sequence_length=SEQUENCE_LENGTH,
            stride=3,
            augment_prob=AUGMENT_PROBABILITY,
            data_name="Train"
        )
        val_dataset = VideoSequenceDataset(
            val_video_items,
            img_size=(IMG_HEIGHT, IMG_WIDTH),
            sequence_length=SEQUENCE_LENGTH,
            stride=3,
            augment_prob=0.0,  # 验证集不使用数据增强
            data_name="Val"
        )

        # 检查数据集是否为空
        if len(train_dataset) == 0:
            raise ValueError(f"训练数据集为空，请检查训练视频数据是否存在")
        if len(val_dataset) == 0:
            raise ValueError(f"验证数据集为空，请检查验证视频数据是否存在")
        if len(test_dataset) == 0:
            raise ValueError(f"测试数据集为空，请检查测试视频数据是否存在")

        print(f"训练数据集大小: {len(train_dataset)} 个序列")
        print(f"验证数据集大小: {len(val_dataset)} 个序列")
        print(f"测试数据集大小: {len(test_dataset)} 个序列")

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        # 使用改进的ConvLSTM模型（包含回归分支）
        model = PupilTrackingConvLSTM(
            input_dim=1,
            hidden_dim=HIDDEN_DIM,
            kernel_size=(3, 3),
            num_layers=1,
        ).to(DEVICE)

        criterion = EuclideanDistanceLoss(weight=1.0)
        heatmap_criterion = HeatmapLoss(IMG_HEIGHT, IMG_WIDTH)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        # 早停机制相关参数
        best_val_loss = float('inf')
        patience = EARLY_STOPPING_PATIENCE
        patience_counter = 0
        min_delta = EARLY_STOPPING_MIN_DELTA

        train_losses = []
        val_losses = []
        lr_values = []

        for epoch in range(1, EPOCHS+1):
            # 训练阶段
            model.train()
            train_loss = 0.0

            train_pbar = tqdm(train_loader, desc=f"Fold {fold+1}/{k_folds} - Epoch {epoch}/{EPOCHS} - Training ", leave=True)
            for data, targets in train_pbar:
                data, targets = data.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()

                preds, seg_output = model(data, return_segmentation=True)

                coord_loss, _ = criterion(pred_coords=preds, gt_coords=targets)
                hm_loss = heatmap_criterion(seg_output, targets)
                loss = coord_loss + HEATMAP_LOSS_WEIGHT * hm_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
                train_loss += loss.item() * data.size(0)

                train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})

            train_loss /= len(train_loader.dataset)

            # 验证阶段
            val_loss = validate_model(epoch, model, val_loader, criterion, DEVICE, fold, heatmap_criterion)
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            lr_values.append(current_lr)
            print(f"Fold {fold+1} - Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}")

            # 早停机制
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                fold_model_path = f"_fold_{fold+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, fold_model_path)
                print(f"  -> Saved best model for fold {fold+1} (val_loss={val_loss:.6f})\n")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> No improvement in validation loss for {patience_counter}/{patience} epochs\n")

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch} for fold {fold+1}!")
                print(f"Best validation loss was {best_val_loss:.6f}")
                break

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

        # 绘制并显示学习率变化曲线
        plt.figure(figsize=(10, 4))
        plt.plot(epochs_range, lr_values, 'g-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title(f'Fold {fold+1}/{k_folds} - Learning Rate Schedule (Cosine)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        lr_save_path = f"lr_curve_fold_{fold+1}.png"
        plt.savefig(lr_save_path, dpi=150)
        print(f"LR curve saved to: {lr_save_path}")
        plt.show()

        # 加载最佳模型进行测试（在固定的测试集上）
        print(f"\nLoading best model for fold {fold+1} testing...")
        fold_model_path = f"_fold_{fold+1}.pth"
        checkpoint = torch.load(fold_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Evaluating model on fixed test set...")
        test_metrics = evaluate_model(model, test_loader, DEVICE, heatmap_criterion)

        print(f"\nFold {fold+1} 固定测试集性能指标:")
        print(f"  平均损失: {test_metrics['avg_loss']:.6f}")
        print(f"  平均距离误差: {test_metrics['avg_distance_error']:.2f} 像素")
        print(f"  中位数距离误差: {test_metrics['median_distance_error']:.2f} 像素")
        print(f"  RMSE: {test_metrics['rmse']:.2f} 像素")
        print(f"  误差标准差: {test_metrics['std_error']:.2f} 像素")

        fold_results.append(test_metrics)

        # 释放当前 Fold 的数据集和 DataLoader，避免内存累积
        del train_dataset, val_dataset, train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
    print(f"测试集占总视频的约 {len(test_video_items)/len(all_video_items)*100:.1f}%")


if __name__ == "__main__":
    main()
