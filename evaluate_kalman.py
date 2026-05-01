"""
自适应卡尔曼滤波器 — 独立验证脚本。

生成包含注视、平滑追踪、扫视的合成瞳孔轨迹，
加入观测噪声后分别用标准卡尔曼和自适应卡尔曼进行滤波，
对比两者的平滑效果和响应速度。
"""
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import RandomWalkKalmanTracker, AdaptiveKalmanTracker


def generate_pupil_trajectory(n_frames=600):
    """生成包含多种运动模式的合成瞳孔轨迹。

    返回 (ground_truth, noisy_observations)，均为归一化坐标 [0,1]。
    轨迹模式: 注视 → 平滑追踪 → 扫视 → 注视 → 扫视 → 平滑追踪
    """
    np.random.seed(42)
    t = np.arange(n_frames)
    gt = np.zeros((n_frames, 2))

    # 轨迹参数定义 (start_frame, end_frame, mode)
    segments = [
        (0,   100, 'fixation',   0.50, 0.50, 0.002),       # 注视中心
        (100, 250, 'pursuit',    0.50, 0.50, 0.08),        # 平滑追踪 → 圆形轨迹
        (250, 251, 'saccade',    0.70, 0.30, 0.0),         # 扫视 (跳变)
        (251, 350, 'fixation',   0.70, 0.30, 0.003),       # 注视新位置
        (350, 351, 'saccade',    0.35, 0.65, 0.0),         # 再次扫视
        (351, 500, 'pursuit',    0.35, 0.65, 0.06),        # 平滑追踪 → 椭圆轨迹
        (500, 600, 'fixation',   0.50, 0.50, 0.002),       # 回到中心注视
    ]

    for seg in segments:
        start, end, mode, base_x, base_y, amp = seg
        idx = slice(start, end)
        seg_t = t[idx] - start
        seg_len = end - start

        if mode == 'fixation':
            # 微小的随机漂移 + 生理震颤
            drift = 0.001 * np.sin(2 * np.pi * seg_t / 50)[:, np.newaxis] * np.array([1, 0.7])
            tremor = np.random.randn(seg_len, 2) * 0.001
            gt[idx] = np.array([base_x, base_y]) + drift + tremor

        elif mode == 'pursuit':
            # 圆形或椭圆形的平滑追踪
            if base_x < 0.5:
                # 顺时针圆
                angle = seg_t / seg_len * 2 * np.pi * 1.5
                gt[idx, 0] = base_x + amp * np.cos(angle)
                gt[idx, 1] = base_y + amp * np.sin(angle) * 0.6
            else:
                # 逆时针椭圆
                angle = seg_t / seg_len * 2 * np.pi * 2.0
                gt[idx, 0] = base_x + amp * 0.7 * np.cos(angle)
                gt[idx, 1] = base_y + amp * np.sin(angle)
            # 叠加轻微噪声模拟追踪误差
            gt[idx] += np.random.randn(seg_len, 2) * 0.0015

        elif mode == 'saccade':
            # 快速跳变（单帧完成，线性插值模拟极短过渡）
            start_pt = np.array([base_x, base_y])
            # 实际跳变方向由前后段决定，这里简化为直接移动
            if start < n_frames - 1:
                # 线性过渡用3帧完成
                n_transition = min(3, end - start)
                for i in range(n_transition):
                    alpha = i / max(n_transition - 1, 1)
                    gt[start + i] = start_pt * (1 - alpha) + gt[start + i]
                if end > start + n_transition:
                    gt[start + n_transition:end] = gt[start + n_transition:end]

    # 扫视段由前后段自动决定，简化处理：直接用目标点
    for seg in segments:
        start, end, mode, base_x, base_y, amp = seg
        if mode == 'saccade':
            gt[start:end] = np.array([base_x, base_y])

    # 确保轨迹在 [0, 1] 内
    gt = np.clip(gt, 0.01, 0.99)

    # 添加观测噪声
    obs_noise_std = 0.008  # 观测噪声标准差 (归一化坐标)
    noisy = gt + np.random.randn(n_frames, 2) * obs_noise_std
    noisy = np.clip(noisy, 0.0, 1.0)

    return gt, noisy


def compute_metrics(filtered, ground_truth):
    """计算滤波质量指标。"""
    n = min(len(filtered), len(ground_truth))
    f = np.array(filtered[:n])
    g = np.array(ground_truth[:n])

    mse = np.mean(np.sum((f - g) ** 2, axis=1))
    rmse = np.sqrt(mse)

    dist = np.sqrt(np.sum((f - g) ** 2, axis=1))
    median_err = np.median(dist)

    # Jerk (平滑度指标): 加速度的方差
    jerk = np.mean(np.diff(f, n=2, axis=0) ** 2)

    return {'mse': mse, 'rmse': rmse, 'median_err': median_err, 'jerk': jerk}


def compute_speed(positions):
    """计算每帧的瞬时速度 (归一化坐标/帧)。"""
    pos = np.array(positions)
    if len(pos) < 2:
        return np.zeros(len(pos))
    speed = np.zeros(len(pos))
    speed[1:] = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
    speed[0] = speed[1]
    return speed


def main():
    print("生成合成瞳孔轨迹 (600帧, 包含注视/平滑追踪/扫视)...")
    gt, noisy = generate_pupil_trajectory()

    gt_speed = compute_speed(gt)

    # ── 标准卡尔曼滤波 ──
    print("运行标准卡尔曼滤波器...")
    kf_std = RandomWalkKalmanTracker(process_noise=1e-1, measurement_noise=5e-2)
    std_result = []
    for obs in noisy:
        filtered, _ = kf_std.update(obs)
        std_result.append(filtered)
    std_metrics = compute_metrics(std_result, gt)

    # ── 自适应卡尔曼滤波 ──
    print("运行自适应卡尔曼滤波器...")
    kf_adp = AdaptiveKalmanTracker(
        base_process_noise=1e-2,
        measurement_noise=5e-2,
        speed_midpoint=0.015,
        vel_sensitivity=100.0,
        min_q_scale=0.05,
        max_q_scale=25.0,
        ema_alpha=0.4,
    )
    adp_result = []
    adp_speeds = []
    adp_q_scales = []
    for obs in noisy:
        filtered, info = kf_adp.update(obs)
        adp_result.append(filtered)
        adp_speeds.append(info['speed'])
        adp_q_scales.append(info['q_scale'])
    adp_metrics = compute_metrics(adp_result, gt)
    adp_speeds = np.array(adp_speeds)
    adp_q_scales = np.array(adp_q_scales)

    # ── 打印对比 ──
    print(f"\n{'='*55}")
    print(f"{'指标':<20} {'标准卡尔曼':>15} {'自适应卡尔曼':>15}")
    print(f"{'-'*55}")
    print(f"{'MSE':<20} {std_metrics['mse']:>15.6f} {adp_metrics['mse']:>15.6f}")
    print(f"{'RMSE':<20} {std_metrics['rmse']:>15.6f} {adp_metrics['rmse']:>15.6f}")
    print(f"{'Median Error':<20} {std_metrics['median_err']:>15.6f} {adp_metrics['median_err']:>15.6f}")
    print(f"{'Jerk (平滑度)':<20} {std_metrics['jerk']:>15.8f} {adp_metrics['jerk']:>15.8f}")
    print(f"{'='*55}")

    # ── 可视化 ──
    _, axes = plt.subplots(3, 2, figsize=(16, 12))

    # (0,0) 2D 轨迹对比
    ax = axes[0, 0]
    ax.plot(gt[:, 0], gt[:, 1], 'k-', linewidth=1.5, alpha=0.6, label='Ground Truth')
    ax.plot(noisy[:, 0], noisy[:, 1], '.', color='gray', markersize=2, alpha=0.5, label='Noisy')
    ax.plot([r[0] for r in std_result], [r[1] for r in std_result],
            'b-', linewidth=1.2, label='Standard KF')
    ax.plot([r[0] for r in adp_result], [r[1] for r in adp_result],
            'r-', linewidth=1.2, label='Adaptive KF')
    ax.set_xlabel('X (norm)')
    ax.set_ylabel('Y (norm)')
    ax.set_title('2D Trajectory Comparison')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # (0,1) X 坐标时序
    ax = axes[0, 1]
    frames = np.arange(len(gt))
    ax.plot(frames, gt[:, 0], 'k-', linewidth=1.2, alpha=0.6, label='Ground Truth')
    ax.plot(frames, [r[0] for r in std_result], 'b-', linewidth=1, label='Standard KF')
    ax.plot(frames, [r[0] for r in adp_result], 'r-', linewidth=1, label='Adaptive KF')
    ax.set_xlabel('Frame')
    ax.set_ylabel('X (norm)')
    ax.set_title('X Coordinate Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) 逐帧误差对比
    ax = axes[1, 0]
    std_err = np.sqrt(np.sum((np.array(std_result) - gt) ** 2, axis=1))
    adp_err = np.sqrt(np.sum((np.array(adp_result) - gt) ** 2, axis=1))
    ax.plot(frames, std_err, 'b-', linewidth=0.8, alpha=0.7, label='Standard KF Error')
    ax.plot(frames, adp_err, 'r-', linewidth=0.8, alpha=0.7, label='Adaptive KF Error')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Euclidean Error (norm)')
    ax.set_title('Per-Frame Tracking Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) 误差分布直方图
    ax = axes[1, 1]
    ax.hist(std_err, bins=40, alpha=0.5, color='blue', label='Standard KF')
    ax.hist(adp_err, bins=40, alpha=0.5, color='red', label='Adaptive KF')
    ax.axvline(np.mean(std_err), color='blue', linestyle='--', linewidth=1.5)
    ax.axvline(np.mean(adp_err), color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Error (norm)')
    ax.set_ylabel('Count')
    ax.set_title('Error Distribution')
    ax.legend(fontsize=8)

    # (2,0) 速度估计 & Q 缩放
    ax = axes[2, 0]
    ax2 = ax.twinx()
    ax.plot(frames, gt_speed, 'k-', linewidth=0.8, alpha=0.4, label='GT Speed')
    ax.plot(frames, adp_speeds, color='orange', linewidth=0.8, label='Est. Speed (EMA)')
    ax2.plot(frames, adp_q_scales, 'g-', linewidth=1.2, label='Q Scale Factor')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Speed (norm/frame)')
    ax2.set_ylabel('Q Scale Factor', color='green')
    ax.set_title('Adaptive Speed Estimation & Q Scaling')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # (2,1) 局部放大: 扫视区域
    ax = axes[2, 1]
    zoom_start, zoom_end = 240, 370
    zoom_frames = frames[zoom_start:zoom_end]
    ax.plot(zoom_frames, gt[zoom_start:zoom_end, 0], 'k-', linewidth=2, alpha=0.7, label='GT')
    ax.plot(zoom_frames, noisy[zoom_start:zoom_end, 0], '.', color='gray',
            markersize=3, alpha=0.5, label='Noisy')
    ax.plot(zoom_frames, [std_result[i][0] for i in range(zoom_start, zoom_end)],
            'b-', linewidth=1.5, label='Standard KF')
    ax.plot(zoom_frames, [adp_result[i][0] for i in range(zoom_start, zoom_end)],
            'r-', linewidth=1.5, label='Adaptive KF')
    ax.set_xlabel('Frame')
    ax.set_ylabel('X (norm)')
    ax.set_title('Zoom: Saccade Region (frames 240-370)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('adaptive_kalman_comparison.png', dpi=150, bbox_inches='tight')
    print("\n对比图已保存至 adaptive_kalman_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
