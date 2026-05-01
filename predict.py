import cv2
import torch
import numpy as np
import csv
from model import PupilTrackingConvLSTM
from config import DEVICE, IMG_HEIGHT, IMG_WIDTH, SEQUENCE_LENGTH
from kalman_filter import AdaptiveKalmanTracker
from config import *
os.environ['PYTHONIOENCODING'] = 'utf-8'

def load_model_weights(model, model_path):
    """
    加载模型权重，处理保存格式不同的情况
    """
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # 检查是否是包含额外训练信息的检查点
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # 加载训练保存的完整状态
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        elif 'state_dict' in checkpoint:
            # 某些保存方式可能使用state_dict键
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            # 尝试直接加载整个字典作为state_dict
            model.load_state_dict(checkpoint, strict=False)
    else:
        # 直接加载checkpoint（如果是纯权重）
        model.load_state_dict(checkpoint, strict=True)
    
    return model

def compute_kalman_metrics(raw_coords, filtered_coords, ground_truth=None):
    """
    评估滤波效果。

    Args:
        raw_coords:      模型原始预测 [(x, y), ...]
        filtered_coords: 滤波后 [(x, y), ...]
        ground_truth:    真实坐标 [(x, y), ...] 或 None

    Returns:
        dict: 各项指标
    """
    raw = np.array(raw_coords)
    flt = np.array(filtered_coords)
    n = len(raw)

    # ── 平滑度: Jerk (二阶差分的均方) ──
    raw_jerk = np.mean(np.diff(raw[:, 0], n=2)**2 + np.diff(raw[:, 1], n=2)**2)
    flt_jerk = np.mean(np.diff(flt[:, 0], n=2)**2 + np.diff(flt[:, 1], n=2)**2)
    smoothness_gain = (1 - flt_jerk / raw_jerk) * 100 if raw_jerk > 0 else 0

    # ── 偏差: 滤波偏离原始预测的程度 ──
    deviation = np.abs(flt - raw)
    mad = np.mean(deviation[:, 0] + deviation[:, 1])  # Mean Absolute Deviation
    max_dev = np.max(deviation[:, 0] + deviation[:, 1])

    # ── 方差缩减率 ──
    raw_var = np.var(raw[:, 0]) + np.var(raw[:, 1])
    flt_var = np.var(flt[:, 0]) + np.var(flt[:, 1])
    var_ratio = flt_var / raw_var if raw_var > 0 else 1.0

    print("=" * 58)
    print("           卡尔曼滤波效果评估")
    print("=" * 58)
    print(f"  总帧数:            {n}")
    print(f"  ─────────── 平滑度 ───────────")
    print(f"  Jerk (原始预测):   {raw_jerk:.2f}")
    print(f"  Jerk (滤波后):     {flt_jerk:.2f}")
    print(f"  平滑度提升:        {smoothness_gain:+.1f}%")
    print(f"  ─────────── 偏差 ───────────")
    print(f"  平均绝对偏差(MAD):  {mad:.2f} px")
    print(f"  最大绝对偏差:       {max_dev:.2f} px")
    print(f"  方差缩减率:         {var_ratio:.3f} (1.0=无变化)")

    # ── 准确度 (仅当有 ground truth) ──
    if ground_truth is not None:
        gt = np.array(ground_truth)
        min_len = min(len(raw), len(gt))
        raw_mse = np.mean(np.sum((raw[:min_len] - gt[:min_len])**2, axis=1))
        flt_mse = np.mean(np.sum((flt[:min_len] - gt[:min_len])**2, axis=1))
        mse_change = (1 - flt_mse / raw_mse) * 100 if raw_mse > 0 else 0
        print(f"  ─────────── 准确度 ───────────")
        print(f"  MSE (原始预测):    {raw_mse:.2f}")
        print(f"  MSE (滤波后):      {flt_mse:.2f}")
        print(f"  准确度变化:        {mse_change:+.1f}%  ({'提升' if mse_change > 0 else '下降'})")

    # ── 综合判读 ──
    print(f"  ─────────── 判读 ───────────")
    if ground_truth is not None:
        if smoothness_gain > 30 and mse_change > 0:
            print("  ✓ 滤波有效: 轨迹更平滑且更准确")
        elif smoothness_gain > 30 and mse_change > -5:
            print("  ~ 滤波平滑有效，准确度基本持平")
        elif mse_change < -10:
            print("  ⚠ 准确度明显下降，可能过平滑，建议减小 measurement_noise 或增大 process_noise")
        elif smoothness_gain < 10 and mse_change < 0:
            print("  ⚠ 平滑不足且准确度下降，滤波参数需重新调优")
        else:
            print("  ~ 效果一般: 可尝试调整噪声参数")
    else:
        if smoothness_gain > 30 and mad < 5:
            print("  ✓ 滤波有效: 轨迹明显平滑，且未显著偏离原始值")
        elif smoothness_gain > 30 and mad > 10:
            print("  ⚠ 可能过平滑: 偏离原始值较大，建议减小 measurement_noise 或增大 process_noise")
        elif smoothness_gain < 10:
            print("  ⚠ 平滑效果不足: 建议增大 measurement_noise 或减小 process_noise")
        else:
            print("  ~ 效果一般: 可尝试调整噪声参数")

    print("=" * 58)

    return {
        "raw_jerk": raw_jerk, "flt_jerk": flt_jerk,
        "smoothness_gain": smoothness_gain,
        "mad": mad, "max_dev": max_dev, "var_ratio": var_ratio,
    }


def load_ground_truth(gt_path, orig_size=None):
    """
    加载真实坐标文件。支持三种格式:

      1. TXT: x y (空格分隔，每行两个数，无 frame 列，无表头)
      2. CSV: frame,x,y (归一化坐标 0~1)
      3. CSV: frame,x,y (像素坐标)

    程序按文件扩展名自动选择解析方式。

    Returns:
        list of (x_pixel, y_pixel)
    """
    gt_data = []

    if gt_path.endswith(".txt"):
        with open(gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    gt_data.append((float(parts[0]), float(parts[1])))
    else:
        with open(gt_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    gt_data.append((float(row[1]), float(row[2])))

    gt = np.array(gt_data)

    # 自动判断坐标类型: 若所有值都在 [0,1] 则是归一化坐标
    if orig_size is not None and np.all(gt >= 0) and np.all(gt <= 1.0):
        gt[:, 0] *= orig_size[0]
        gt[:, 1] *= orig_size[1]

    return [(x, y) for x, y in gt]

def predict_and_annotate_video(video_path, model_path, output_path, orig_size=None,
                               confidence_threshold=None, ground_truth_path=None):
    """
    使用训练模型对视频逐帧预测并在原视频上标注瞳孔位置

    Args:
        video_path: 输入视频路径
        model_path: 模型权重路径
        output_path: 输出视频路径
        orig_size: 原始视频尺寸 (width, height)，如果为None则自动获取
        confidence_threshold: 置信度阈值（如果模型输出置信度分数）
        ground_truth_path: 真实坐标CSV路径 (可选)，用于评估准确度
    """
    model = PupilTrackingConvLSTM().to(DEVICE)
    model = load_model_weights(model, model_path)
    model.eval()
    
    # 初始化卡尔曼滤波器
    kalman_filter = AdaptiveKalmanTracker()

    # 打开输入视频
    cap = cv2.VideoCapture(video_path)
    
    # 自动获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果未指定原始尺寸，则使用视频的实际尺寸
    if orig_size is None:
        orig_size = (original_width, original_height)
    
    # 创建视频写入对象 - 输出保持原始尺寸
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (original_width, original_height))

    # 加载 ground truth (如果有)
    ground_truth = None
    if ground_truth_path is not None and os.path.exists(ground_truth_path):
        ground_truth = load_ground_truth(ground_truth_path, orig_size)
        print(f"已加载真实坐标: {len(ground_truth)} 帧")

    frame_count = 0
    print(f"开始处理视频: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps}")
    print(f"原始视频尺寸: {original_width}x{original_height}")
    print(f"模型输入尺寸: {IMG_WIDTH}x{IMG_HEIGHT}")

    # 收集坐标用于最终评估
    all_raw_coords = []
    all_filtered_coords = []

    # 存储帧缓存以构建序列
    frame_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 保存原始帧用于输出
        original_frame = frame.copy()
        
        # 将输入帧调整为模型训练时的尺寸
        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            
        # 预测瞳孔位置
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        # 注意：这里不需要再次resize，因为我们已经resize到了模型训练的尺寸
        resized = gray  # 已经是IMG_WIDTH x IMG_HEIGHT
        frame_tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0  # shape: (1, 1, H, W)
        frame_tensor = frame_tensor.to(DEVICE)
        
        # 将当前帧添加到缓冲区
        frame_buffer.append(frame_tensor)
        
        # 如果缓冲区大小小于序列长度，使用重复的最后一帧填充
        if len(frame_buffer) < SEQUENCE_LENGTH:
            # 用重复的帧填充直到达到所需长度
            recent_frames = frame_buffer + [frame_buffer[-1]] * (SEQUENCE_LENGTH - len(frame_buffer))
            padded_sequence = torch.cat(recent_frames, dim=1)  # Concatenate along channel-like dimension
        else:
            # 取最后SEQUENCE_LENGTH帧并连接
            recent_frames = frame_buffer[-SEQUENCE_LENGTH:]
            padded_sequence = torch.cat(recent_frames, dim=1)  # Concatenate along channel-like dimension
        
        # 重塑为正确的5D张量 (batch, seq_len, channels, height, width)
        batch_size, flat_channels, height, width = padded_sequence.shape
        reshaped_sequence = padded_sequence.view(batch_size, SEQUENCE_LENGTH, 1, height, width)
        
        with torch.no_grad():
            pred_norm = model(reshaped_sequence).cpu().numpy()
            # 确保pred_norm是期望的形状并提取坐标
            if pred_norm.ndim > 1:
                pred_norm = pred_norm[0]  # 获取第一个样本的预测结果
            
        # 确保pred_norm是2维向量 [x, y]
        if pred_norm.size == 2:
            pred_x, pred_y = pred_norm.flatten()[:2]
        else:
            # 如果模型输出格式不同，根据实际情况调整
            pred_x = pred_norm[0] if len(pred_norm) > 0 else 0.5
            pred_y = pred_norm[1] if len(pred_norm) > 1 else 0.5

        # 转换归一化坐标到原始视频尺寸
        raw_x_pixel = int(float(pred_x) * orig_size[0])
        raw_y_pixel = int(float(pred_y) * orig_size[1])
        
        # 应用卡尔曼滤波器平滑预测结果
        raw_coords = [raw_x_pixel, raw_y_pixel]
        filtered_result = kalman_filter.update(raw_coords)
        # 处理返回值，确保是标量数值
        if isinstance(filtered_result, tuple):
            filtered_coords, _ = filtered_result  # 如果返回位置和速度元组
        else:
            filtered_coords = filtered_result   # 如果只返回位置
        
        # 确保filtered_coords是数组格式并提取前两个元素
        filtered_coords = np.asarray(filtered_coords)
        x_pixel, y_pixel = int(filtered_coords[0]), int(filtered_coords[1])

        # 收集坐标用于最终评估
        all_raw_coords.append((raw_x_pixel, raw_y_pixel))
        all_filtered_coords.append((float(filtered_coords[0]), float(filtered_coords[1])))

        # 在原始帧上绘制瞳孔位置
        # 绘制一个圆圈标记瞳孔位置
        cv2.circle(original_frame, (x_pixel, y_pixel), radius=20, color=(0, 255, 0), thickness=2)
        
        # 绘制十字标记
        cv2.line(original_frame, (x_pixel-25, y_pixel), (x_pixel+25, y_pixel), (0, 255, 0), 3)
        cv2.line(original_frame, (x_pixel, y_pixel-25), (x_pixel, y_pixel+25), (0, 255, 0), 3)
        
        # 添加坐标文本
        cv2.putText(original_frame, f'Pupil: ({x_pixel}, {y_pixel})', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 显示原始预测值以作比较
        cv2.putText(original_frame, f'Raw: ({raw_x_pixel}, {raw_y_pixel})', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 添加帧数信息
        cv2.putText(original_frame, f'Frame: {frame_count}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 将处理后的帧写入输出视频
        out.write(original_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧")

    # ── 评估滤波效果 ──
    if len(all_raw_coords) > 2:
        gt = ground_truth[:frame_count] if ground_truth is not None else None
        compute_kalman_metrics(all_raw_coords, all_filtered_coords, gt)

    # 释放资源
    cap.release()
    out.release()
    print(f"视频处理完成! 输出保存至: {output_path}")
    return frame_count

def predict_video_with_coordinates(video_path, model_path, orig_size=None, use_kalman=True,
                                   ground_truth_path=None):
    """
    返回预测坐标但不标注视频
    Args:
        ground_truth_path: 真实坐标CSV路径 (可选)，用于评估准确度
    """
    model = PupilTrackingConvLSTM().to(DEVICE)
    model = load_model_weights(model, model_path)
    model.eval()

    # 加载 ground truth (如果有)
    ground_truth = None
    if ground_truth_path is not None and os.path.exists(ground_truth_path):
        ground_truth = load_ground_truth(ground_truth_path, orig_size)

    # 初始化卡尔曼滤波器
    kalman_filter = None
    if use_kalman:
        kalman_filter = AdaptiveKalmanTracker()

    cap = cv2.VideoCapture(video_path)
    
    # 自动获取视频属性
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果未指定原始尺寸，则使用视频的实际尺寸
    if orig_size is None:
        orig_size = (original_width, original_height)
    
    results = []
    all_raw_coords = []
    all_filtered_coords = []

    # 存储帧缓存以构建序列
    frame_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将输入帧调整为模型训练时的尺寸
        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        # 注意：这里不需要再次resize，因为我们已经resize到了模型训练的尺寸
        resized = gray  # 已经是IMG_WIDTH x IMG_HEIGHT
        frame_tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0  # shape: (1, 1, H, W)
        frame_tensor = frame_tensor.to(DEVICE)
        
        # 将当前帧添加到缓冲区
        frame_buffer.append(frame_tensor)
        
        # 如果缓冲区大小小于序列长度，使用重复的最后一帧填充
        if len(frame_buffer) < SEQUENCE_LENGTH:
            # 用重复的帧填充直到达到所需长度
            recent_frames = frame_buffer + [frame_buffer[-1]] * (SEQUENCE_LENGTH - len(frame_buffer))
            padded_sequence = torch.cat(recent_frames, dim=1)  # Concatenate along channel-like dimension
        else:
            # 取最后SEQUENCE_LENGTH帧并连接
            recent_frames = frame_buffer[-SEQUENCE_LENGTH:]
            padded_sequence = torch.cat(recent_frames, dim=1)  # Concatenate along channel-like dimension
        
        # 重塑为正确的5D张量 (batch, seq_len, channels, height, width)
        batch_size, flat_channels, height, width = padded_sequence.shape
        reshaped_sequence = padded_sequence.view(batch_size, SEQUENCE_LENGTH, 1, height, width)
        
        with torch.no_grad():
            pred_norm = model(reshaped_sequence).cpu().numpy()
            # 确保pred_norm是期望的形状并提取坐标
            if pred_norm.ndim > 1:
                pred_norm = pred_norm[0]  # 获取第一个样本的预测结果
            
        # 确保pred_norm是2维向量 [x, y]
        if pred_norm.size == 2:
            pred_x, pred_y = pred_norm.flatten()[:2]
        else:
            # 如果模型输出格式不同，根据实际情况调整
            pred_x = pred_norm[0] if len(pred_norm) > 0 else 0.5
            pred_y = pred_norm[1] if len(pred_norm) > 1 else 0.5

        raw_x_pixel = float(pred_x) * orig_size[0]
        raw_y_pixel = float(pred_y) * orig_size[1]
        if use_kalman:
            # 应用卡尔曼滤波器
            raw_coords = [raw_x_pixel, raw_y_pixel]
            filtered_result = kalman_filter.update(raw_coords)
            # 处理返回值，确保是标量数值
            if isinstance(filtered_result, tuple):
                filtered_coords, _ = filtered_result  # 如果返回位置和速度元组
            else:
                filtered_coords = filtered_result   # 如果只返回位置
            
            # 确保filtered_coords是数组格式并提取前两个元素
            filtered_coords = np.asarray(filtered_coords)
            results.append((raw_x_pixel, raw_y_pixel, float(filtered_coords[0]), float(filtered_coords[1])))
            all_raw_coords.append((raw_x_pixel, raw_y_pixel))
            all_filtered_coords.append((float(filtered_coords[0]), float(filtered_coords[1])))
        else:
            results.append((raw_x_pixel, raw_y_pixel, raw_x_pixel, raw_y_pixel))
            all_raw_coords.append((raw_x_pixel, raw_y_pixel))
            all_filtered_coords.append((raw_x_pixel, raw_y_pixel))

    # ── 评估滤波效果 ──
    if use_kalman and len(all_raw_coords) > 2:
        gt = ground_truth[:len(all_raw_coords)] if ground_truth is not None else None
        compute_kalman_metrics(all_raw_coords, all_filtered_coords, gt)

    cap.release()
    return results

if __name__ == "__main__":
    # ── 配置 ──
    input_video_path = os.path.join(LPW_ROOT, "5\\10.avi")
    # input_video_path = "D:\\vedio2.mp4"
    model_path = "checkpoints.pth"
    output_video_path = "annotated_pupil_tracking.mp4"
    use_kalman_filter = True    # 是否使用卡尔曼滤波器
    ground_truth_path = os.path.join(LPW_ROOT, "5\\10.txt")  # 真实坐标: x y 空格分隔

    import os
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请确保模型文件存在后再运行此脚本")
        exit(1)

    if not os.path.exists(input_video_path):
        print(f"视频文件不存在: {input_video_path}")
        print("请确保视频文件存在后再运行此脚本")
        exit(1)

    # 处理视频并添加标注
    total_frames = predict_and_annotate_video(
        video_path=input_video_path,
        model_path=model_path,
        output_path=output_video_path,
        orig_size=(640, 480),
        ground_truth_path=ground_truth_path,
    )

    # 获取坐标数据
    coords = predict_video_with_coordinates(
        input_video_path, model_path, ground_truth_path=ground_truth_path)

    # 打印部分坐标信息
    for i, (raw_x, raw_y, filtered_x, filtered_y) in enumerate(coords[:10]):
        print(f"Frame {i}: Raw({raw_x:.1f}, {raw_y:.1f}), Filtered({filtered_x:.1f}, {filtered_y:.1f})")

    # 保存坐标到CSV文件
    with open("predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "raw_x", "raw_y", "filtered_x", "filtered_y"])
        for i, (raw_x, raw_y, filtered_x, filtered_y) in enumerate(coords):
            writer.writerow([i, f"{raw_x:.2f}", f"{raw_y:.2f}", f"{filtered_x:.2f}", f"{filtered_y:.2f}"])

    print("预测坐标已保存到 predictions.csv")