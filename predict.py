import cv2
import torch
import numpy as np
from model import PupilTrackingConvLSTM
from config import DEVICE, IMG_HEIGHT, IMG_WIDTH, SEQUENCE_LENGTH
from kalman_filter import AdaptiveKalmanFilter
from config import *

def predict_and_annotate_video(video_path, model_path, output_path, orig_size=None, confidence_threshold=None):
    """
    使用训练模型对视频逐帧预测并在原视频上标注瞳孔位置
    
    Args:
        video_path: 输入视频路径
        model_path: 模型权重路径
        output_path: 输出视频路径
        orig_size: 原始视频尺寸 (width, height)，如果为None则自动获取
        confidence_threshold: 置信度阈值（如果模型输出置信度分数）
    """
    model = PupilTrackingConvLSTM().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 初始化卡尔曼滤波器
    kalman_filter = AdaptiveKalmanFilter()

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

    frame_count = 0
    print(f"开始处理视频: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps}")
    print(f"原始视频尺寸: {original_width}x{original_height}")
    print(f"模型输入尺寸: {IMG_WIDTH}x{IMG_HEIGHT}")

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
            pred_norm = model(reshaped_sequence).cpu().numpy()[0]
            
        # 转换归一化坐标到原始视频尺寸
        raw_x_pixel = int(pred_norm[0] * orig_size[0])
        raw_y_pixel = int(pred_norm[1] * orig_size[1])
        
        # 应用卡尔曼滤波器平滑预测结果
        raw_coords = [raw_x_pixel, raw_y_pixel]
        filtered_coords = kalman_filter.update(raw_coords)
        x_pixel, y_pixel = int(filtered_coords[0]), int(filtered_coords[1])
        
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

    # 释放资源
    cap.release()
    out.release()
    print(f"视频处理完成! 输出保存至: {output_path}")
    return frame_count

def predict_video_with_coordinates(video_path, model_path, orig_size=None, use_kalman=True):
    """
    保留原始功能：返回预测坐标但不标注视频
    """
    model = PupilTrackingConvLSTM().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    
    # 初始化卡尔曼滤波器
    kalman_filter = None
    if use_kalman:
        kalman_filter = AdaptiveKalmanFilter()

    cap = cv2.VideoCapture(video_path)
    
    # 自动获取视频属性
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 如果未指定原始尺寸，则使用视频的实际尺寸
    if orig_size is None:
        orig_size = (original_width, original_height)
    
    results = []
    
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
            pred_norm = model(reshaped_sequence).cpu().numpy()[0]
        raw_x_pixel = pred_norm[0] * orig_size[0]
        raw_y_pixel = pred_norm[1] * orig_size[1]
        if use_kalman:
            # 应用卡尔曼滤波器
            raw_coords = [raw_x_pixel, raw_y_pixel]
            filtered_coords = kalman_filter.update(raw_coords)
            results.append((raw_x_pixel, raw_y_pixel, filtered_coords[0], filtered_coords[1]))
        else:
            results.append((raw_x_pixel, raw_y_pixel, raw_x_pixel, raw_y_pixel))  # 当不使用卡尔曼滤波时，也返回原始坐标两次
    
    cap.release()
    return results

if __name__ == "__main__":
    # 标注视频并保存
    input_video_path = os.path.join(LPW_ROOT, "4\\2.avi")
    # input_video_path = "D:\\vedio2.mp4"
    model_path = "checkpoint.pth"
    output_video_path = "annotated_pupil_tracking.mp4"
    use_kalman_filter = True    # 是否使用卡尔曼滤波器
    
    # 检查模型文件是否存在
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
        orig_size=(640, 480)  # 根据实际视频尺寸调整
    )
    
    # 如果想获取坐标数据，可以使用原来的函数
    coords = predict_video_with_coordinates(input_video_path, model_path)
    
    # 打印部分坐标信息
    for i, (raw_x, raw_y, filtered_x, filtered_y) in enumerate(coords[:10]):  # 只打印前10帧
        print(f"Frame {i}: Raw({raw_x:.1f}, {raw_y:.1f}), Filtered({filtered_x:.1f}, {filtered_y:.1f})")
    
    # 保存坐标到CSV文件
    import csv
    with open("predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "raw_x", "raw_y", "filtered_x", "filtered_y"])
        for i, (raw_x, raw_y, filtered_x, filtered_y) in enumerate(coords):
            writer.writerow([i, f"{raw_x:.2f}", f"{raw_y:.2f}", f"{filtered_x:.2f}", f"{filtered_y:.2f}"])
   
    print("预测坐标已保存到 predictions.csv")