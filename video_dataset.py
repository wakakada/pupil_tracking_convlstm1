import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class VideoSequenceDataset(Dataset):
    def __init__(self, video_items, img_size=(45, 60), sequence_length=10, stride=1, augment_prob=0.5, data_name=""):
        """
        video_items: list of (video_path, label_path, video_name) tuples
                     video_name 为显示用名称，如 "5/10.avi"（受试者/视频文件）
        img_size: (height, width) 模型输入尺寸
        sequence_length: 序列长度（LSTM需要的帧数）
        stride: 序列之间的步长
        augment_prob: 数据增强的概率
        data_name: 数据集名称
        """
        self.img_h, self.img_w = img_size
        self.sequence_length = sequence_length
        self.stride = stride
        self.augment_prob = augment_prob
        self.data_name = data_name

        # 每个视频存为一个 (frames_uint8, labels_float32) 元组
        # frames_uint8: (N, H, W)  uint8  —— 4.8KB/帧 vs 原来的 19KB/帧
        # labels_array:  (N, 2)   float32 —— 已归一化坐标
        self.video_data = []
        # 序列索引: (video_idx, start_frame)，不在内存中缓存序列副本
        self.sequence_indices = []

        print(f"正在加载{self.data_name}数据集，共 {len(video_items)} 个视频...")

        for video_path, label_path, video_name in tqdm(video_items, desc="Processing Videos"):
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Cannot open video {video_path}, skipping...")
                continue
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames < self.sequence_length:
                print(f"Warning: Video {video_name} has only {total_frames} frames, less than required sequence length {self.sequence_length}, skipping...")
                cap.release()
                continue

            # 读取所有帧为 uint8 灰度图
            frames_list = []
            for _ in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (self.img_w, self.img_h))
                frames_list.append(resized)
            cap.release()

            if len(frames_list) < self.sequence_length:
                print(f"Warning: Video {video_name} read only {len(frames_list)} valid frames, skipping...")
                continue

            frames_uint8 = np.stack(frames_list, axis=0)  # (N, H, W) uint8

            # 读取标签并归一化
            with open(label_path, 'r') as f:
                label_lines = [line.strip() for line in f.readlines()]

            labels_list = []
            n_valid = len(frames_list)
            for idx in range(n_valid):
                if idx >= len(label_lines):
                    break
                parts = label_lines[idx].split()
                x, y = float(parts[0]), float(parts[1])
                labels_list.append([x / orig_w, y / orig_h])

            labels_array = np.array(labels_list, dtype=np.float32)  # (N, 2)

            video_idx = len(self.video_data)
            self.video_data.append((frames_uint8, labels_array))

            # 仅记录序列起止索引，不在内存中构建序列张量
            for start_idx in range(0, n_valid - self.sequence_length + 1, self.stride):
                self.sequence_indices.append((video_idx, start_idx))

        print(f"{self.data_name}数据集加载完成，共创建了 {len(self.sequence_indices)} 个序列样本")

    def __len__(self):
        return len(self.sequence_indices)

    def __getitem__(self, idx):
        video_idx, start_frame = self.sequence_indices[idx]
        frames_uint8, labels_array = self.video_data[video_idx]

        # 切片 uint8 视图 → float32 归一化（仅对 12 帧做，几乎无开销）
        seq_slice = frames_uint8[start_frame:start_frame + self.sequence_length]  # (seq_len, H, W) uint8 view
        sequence_tensor = torch.from_numpy(seq_slice).float() / 255.0               # (seq_len, H, W) float32
        sequence_tensor = sequence_tensor.unsqueeze(1)                              # (seq_len, 1, H, W)

        target_label = torch.from_numpy(labels_array[start_frame + self.sequence_length - 1])  # (2,)

        if random.random() < self.augment_prob:
            do_flip = random.random() < 0.5
            sequence_tensor = self._apply_augmentation(sequence_tensor, do_flip)
            if do_flip:
                target_label = target_label.clone()
                target_label[0] = 1.0 - target_label[0]

        return sequence_tensor, target_label

    def _apply_augmentation(self, sequence_tensor, do_flip=False):
        """对序列中的每一帧应用数据增强（翻转在整个序列上一致）"""
        augmented_sequence = sequence_tensor.clone()

        for frame_idx in range(sequence_tensor.size(0)):
            frame = augmented_sequence[frame_idx, 0]  # (H, W)

            if do_flip:
                frame = torch.flip(frame, [1])

            if random.random() < 0.3:
                brightness_factor = random.uniform(0.8, 1.2)
                frame = torch.clamp(frame * brightness_factor, 0.0, 1.0)

            if random.random() < 0.3:
                contrast_factor = random.uniform(0.8, 1.2)
                mean_val = torch.mean(frame)
                frame = torch.clamp((frame - mean_val) * contrast_factor + mean_val, 0.0, 1.0)

            if random.random() < 0.2:
                noise_std = random.uniform(0.0, 0.05)
                noise = torch.randn_like(frame) * noise_std
                frame = torch.clamp(frame + noise, 0.0, 1.0)

            if random.random() < 0.2:
                gamma = random.uniform(0.8, 1.2)
                frame = torch.pow(frame, gamma)

            if random.random() < 0.3:
                dx = random.randint(-4, 4)
                dy = random.randint(-4, 4)
                frame = torch.roll(frame, shifts=(dy, dx), dims=(0, 1))

            if random.random() < 0.1:
                import torchvision.transforms as transforms
                transform = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
                frame = transform(frame.unsqueeze(0)).squeeze(0)

            if random.random() < 0.1:
                h, w = frame.shape
                start_h = random.randint(0, h-5)
                start_w = random.randint(0, w-5)
                rect_h = random.randint(2, 5)
                rect_w = random.randint(2, 5)
                frame[start_h:start_h+rect_h, start_w:start_w+rect_w] = 0.5

            augmented_sequence[frame_idx, 0] = frame

        return augmented_sequence


def build_video_items(lpw_root):
    """
    扫描LPW数据集目录，返回所有视频项的列表。
    每项为 (video_path, label_path, video_name)，其中 video_name 格式为 "受试者/video.avi"。

    Args:
        lpw_root: LPW数据集根目录

    Returns:
        list of (video_path, label_path, video_name)
    """
    video_items = []

    subject_dirs = [d for d in os.listdir(lpw_root) if os.path.isdir(os.path.join(lpw_root, d)) and d.isdigit()]
    subject_dirs.sort(key=int)

    for subject_id in subject_dirs:
        subject_path = os.path.join(lpw_root, subject_id)

        video_files = [f for f in os.listdir(subject_path) if f.lower().endswith('.avi')]

        for vf in video_files:
            video_path = os.path.join(subject_path, vf)
            label_name = os.path.splitext(vf)[0] + '.txt'
            label_path = os.path.join(subject_path, label_name)

            if os.path.exists(label_path):
                video_name = f"{subject_id}/{vf}"
                video_items.append((video_path, label_path, video_name))
            else:
                print(f"Warning: Label file {label_path} not found, skipping {video_name}")

    return video_items
