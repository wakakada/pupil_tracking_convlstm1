import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class VideoSequenceDataset(Dataset):
    def __init__(self, lpw_root, subject_ids, img_size=(45, 60), sequence_length=10, stride=1, augment_prob=0.5, data_name=""):
        """
        lpw_root: LPW 数据集根目录 (如 ./LPW/)
        subject_ids: 受试者ID列表 (例如 [1, 2, 3, ...])
        img_size: (height, width) 模型输入尺寸
        sequence_length: 序列长度（LSTM需要的帧数）
        stride: 序列之间的步长
        augment_prob: 数据增强的概率
        data_name: 数据集名称
        """
        self.lpw_root = lpw_root
        self.subject_ids = subject_ids
        self.img_h, self.img_w = img_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.cached_sequences = []   # 存储所有 (sequence_tensor, target_tensor)
        self.augment_prob = augment_prob
        self.data_name = data_name

        # 收集所有视频文件路径
        all_video_paths = []
        for subject_id in self.subject_ids:
            subject_path = os.path.join(self.lpw_root, str(subject_id))
            
            if not os.path.isdir(subject_path):
                print(f"Warning: Subject directory {subject_path} does not exist, skipping...")
                continue
                
            # 获取该受试者的所有视频和标签文件
            video_files = [f for f in os.listdir(subject_path) if f.lower().endswith('.avi')]
            
            for vf in video_files:
                video_path = os.path.join(subject_path, vf)
                label_name = os.path.splitext(vf)[0] + '.txt'
                label_path = os.path.join(subject_path, label_name)
                
                if os.path.exists(label_path):
                    all_video_paths.append((video_path, label_path, subject_path))
                else:
                    print(f"Warning: Label file {label_path} not found, skipping {vf}")

        print(f"正在加载{self.data_name}数据集，共 {len(all_video_paths)} 个视频...")
        
        # 使用进度条处理所有视频
        for video_path, label_path, subject_path in tqdm(all_video_paths, desc="Processing Videos"):
            # 获取视频属性（总帧数，原始尺寸）
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            if total_frames < self.sequence_length:
                print(f"Warning: Video {video_path} has only {total_frames} frames, less than required sequence length {self.sequence_length}, skipping...")
                continue

            # 读取所有标签（原始像素坐标）
            with open(label_path, 'r') as f:
                label_lines = [line.strip() for line in f.readlines()]
            
            # 确保标签行数足够
            if len(label_lines) < total_frames:
                print(f"Warning: Label file {label_path} has fewer lines ({len(label_lines)}) than frames ({total_frames}), skipping video")
                continue

            # 创建序列
            frames = []
            labels = []
            
            # 一次性读取所有帧和标签
            cap = cv2.VideoCapture(video_path)
            for idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    print(f"Warning: Cannot read frame {idx} from {video_path}, skipping")
                    continue
                    
                # 处理图像
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (self.img_w, self.img_h))
                normalized = resized.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(normalized).float()  # (H, W)

                # 解析标签
                line = label_lines[idx]
                parts = line.split()
                x, y = float(parts[0]), float(parts[1])
                
                # 归一化坐标
                x_norm = x / orig_w
                y_norm = y / orig_h
                label_tensor = torch.tensor([x_norm, y_norm], dtype=torch.float32)

                frames.append(frame_tensor)
                labels.append(label_tensor)
            
            cap.release()

            # 创建序列
            for start_idx in range(0, len(frames) - self.sequence_length + 1, self.stride):
                sequence_frames = frames[start_idx:start_idx + self.sequence_length]
                sequence_labels = labels[start_idx:start_idx + self.sequence_length]
                
                # 使用序列的最后一帧的标签作为目标
                target_label = sequence_labels[-1]
                
                # 将帧序列堆叠成张量 (seq_len, H, W)
                sequence_tensor = torch.stack(sequence_frames)
                sequence_tensor = sequence_tensor.unsqueeze(1)  # 添加通道维度 -> (seq_len, 1, H, W)
                
                self.cached_sequences.append((sequence_tensor, target_label))
        
        print(f"{self.data_name}数据集加载完成，共创建了 {len(self.cached_sequences)} 个序列样本")

    def __len__(self):
        return len(self.cached_sequences)

    def __getitem__(self, idx):
        sequence_tensor, target_label = self.cached_sequences[idx]
        
        # 应用数据增强
        if random.random() < self.augment_prob:
            sequence_tensor = self._apply_augmentation(sequence_tensor)
        
        return sequence_tensor, target_label   # 返回序列和目标标签

    def _apply_augmentation(self, sequence_tensor):
        """对序列中的每一帧应用数据增强"""
        augmented_sequence = sequence_tensor.clone()
        
        for frame_idx in range(sequence_tensor.size(0)):  # 遍历序列中的每一帧
            frame = augmented_sequence[frame_idx, 0]  # 提取单帧 (H, W)
            
            # 随机翻转
            if random.random() < 0.5:
                frame = torch.flip(frame, [1])  # 水平翻转
                # 如果翻转了图像，需要相应地调整x坐标
                # 注意：由于这是序列数据集，我们不能在这里调整target_label，因为增强只应用于输入
            
            # 随机亮度调整
            if random.random() < 0.3:
                brightness_factor = random.uniform(0.8, 1.2)
                frame = torch.clamp(frame * brightness_factor, 0.0, 1.0)
            
            # 随机对比度调整
            if random.random() < 0.3:
                contrast_factor = random.uniform(0.8, 1.2)
                mean_val = torch.mean(frame)
                frame = torch.clamp((frame - mean_val) * contrast_factor + mean_val, 0.0, 1.0)
            
            # 添加高斯噪声
            if random.random() < 0.2:
                noise_std = random.uniform(0.0, 0.05)
                noise = torch.randn_like(frame) * noise_std
                frame = torch.clamp(frame + noise, 0.0, 1.0)
                
            # 随机伽马变换
            if random.random() < 0.2:
                gamma = random.uniform(0.8, 1.2)
                frame = torch.pow(frame, gamma)
                
             # 随机高斯模糊
            if random.random() < 0.1:
                import torchvision.transforms as transforms
                transform = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                frame = transform(frame.unsqueeze(0)).squeeze(0)
            
            # 随机遮挡
            if random.random() < 0.1:
                h, w = frame.shape
                # 随机选择一个小矩形区域进行遮挡
                start_h = random.randint(0, h-5)
                start_w = random.randint(0, w-5)
                rect_h = random.randint(2, 5)
                rect_w = random.randint(2, 5)
                frame[start_h:start_h+rect_h, start_w:start_w+rect_w] = 0.5  # 用中性灰遮挡
            
            augmented_sequence[frame_idx, 0] = frame  # 存回序列
        
        return augmented_sequence