import numpy as np


class RandomWalkKalmanTracker:
    """随机游走卡尔曼滤波器 — 无运动假设，适合眼球跟踪。

    状态: [x, y] (仅位置，不做速度外推)
    运动模型: x_k = x_{k-1} + w  (w ~ N(0, Q))
    观测模型: z_k = x_k + v      (v ~ N(0, R))

    与 CV 卡尔曼的关键区别: F = I，不做匀速预测，
    跳视时不会按上一帧速度方向"跑偏"。
    """
    def __init__(self, process_noise=1e-1, measurement_noise=5e-2):
        self.F = np.eye(2)
        self.H = np.eye(2)
        self.Q = np.eye(2) * process_noise
        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(2) * 100
        self.x = np.zeros(2)
        self.initialized = False

    def update(self, measurement):
        measurement = np.array(measurement, dtype=np.float64)

        if not self.initialized:
            self.x = measurement.copy()
            self.P = np.eye(2) * 100
            self.initialized = True
            return self.x.copy(), None

        # 预测 (F = I → x_pred = x, P_pred = P + Q)
        x_pred = self.x
        P_pred = self.P + self.Q

        # 更新 (H = I)
        S = P_pred + self.R
        K = P_pred @ np.linalg.inv(S)
        innovation = measurement - x_pred

        self.x = x_pred + K @ innovation
        self.P = (np.eye(2) - K) @ P_pred
        self.P = (self.P + self.P.T) / 2

        return self.x.copy(), None

    def reset(self):
        self.x = np.zeros(2)
        self.P = np.eye(2) * 100
        self.initialized = False


class AdaptiveKalmanTracker:
    """自适应卡尔曼滤波器 — 根据实时运动速度动态调整过程噪声协方差。

    状态: [x, y, vx, vy] (位置 + 速度)
    运动模型: 匀速模型 (CV)，F 按 dt=1 构造
    观测模型: z_k = H @ x_k + v  (只观测位置)

    Q 的自适应策略:
      - 低速 (平滑追踪/注视): Q 缩小 → 侧重平滑，抑制抖动
      - 高速 (扫视/跳视):   Q 放大 → 加速响应，减少滞后

    Q 缩放函数使用 tanh 平滑过渡，避免硬阈值切换。
    """

    def __init__(self,
                 base_process_noise=1e-2,
                 measurement_noise=5e-2,
                 speed_midpoint=0.02,
                 vel_sensitivity=80.0,
                 min_q_scale=0.05,
                 max_q_scale=20.0,
                 ema_alpha=0.3):
        """
        Args:
            base_process_noise: Q 的基础值（位置分量）
            measurement_noise:   观测噪声 R
            speed_midpoint:      速度中点（归一化坐标/帧），scale=0.5*(min+max) 时的速度
            vel_sensitivity:     tanh 过渡的陡峭程度，越大过渡越陡
            min_q_scale:         Q 最小缩放因子（低速时，强平滑）
            max_q_scale:         Q 最大缩放因子（高速时，快响应）
            ema_alpha:           速度估计的指数移动平均系数
        """
        # 状态转移矩阵 (CV 模型, dt=1)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float64)

        # 观测矩阵 — 只观测位置
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float64)

        self.Q_base_pos = base_process_noise
        self.Q_base_vel = base_process_noise * 0.25  # 速度分量基础噪声更小

        self.R = np.eye(2) * measurement_noise
        self.P = np.eye(4) * 100
        self.x = np.zeros(4)
        self.initialized = False

        # 自适应参数
        self.speed_midpoint = speed_midpoint
        self.vel_sensitivity = vel_sensitivity
        self.min_q_scale = min_q_scale
        self.max_q_scale = max_q_scale
        self.ema_alpha = ema_alpha

        # 速度估计状态
        self._ema_speed = 0.0
        self._prev_measurement = None
        self._current_q_scale = 1.0

    def _compute_q_scale(self, speed):
        """tanh 平滑过渡: speed → Q scale factor."""
        x = (speed - self.speed_midpoint) * self.vel_sensitivity
        x = np.clip(x, -5.0, 5.0)
        scale = self.min_q_scale + \
                (self.max_q_scale - self.min_q_scale) * 0.5 * (1.0 + np.tanh(x))
        return scale

    def _build_Q(self, scale):
        """根据缩放因子构造 Q 矩阵。"""
        q_pos = self.Q_base_pos * scale
        q_vel = self.Q_base_vel * scale
        return np.diag([q_pos, q_pos, q_vel, q_vel])

    @property
    def current_q_scale(self):
        return self._current_q_scale

    @property
    def estimated_speed(self):
        return self._ema_speed

    def update(self, measurement):
        measurement = np.array(measurement, dtype=np.float64)

        if not self.initialized:
            self.x[0] = measurement[0]
            self.x[1] = measurement[1]
            self.x[2] = 0.0
            self.x[3] = 0.0
            self.P = np.eye(4) * 100
            self.initialized = True
            self._prev_measurement = measurement.copy()
            return self.x[:2].copy(), {
                'speed': 0.0,
                'q_scale': 1.0,
                'innovation': np.zeros(2)
            }

        # 估计当前速度 (基于原始测量的帧间位移)
        raw_speed = np.linalg.norm(measurement - self._prev_measurement)
        self._ema_speed = (self.ema_alpha * raw_speed +
                           (1 - self.ema_alpha) * self._ema_speed)
        self._prev_measurement = measurement.copy()

        # 根据速度自适应调整 Q
        q_scale = self._compute_q_scale(self._ema_speed)
        self._current_q_scale = q_scale
        Q = self._build_Q(q_scale)

        # 预测
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + Q

        # 更新
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        innovation = measurement - self.H @ x_pred

        self.x = x_pred + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        self.P = (self.P + self.P.T) / 2

        info = {
            'speed': self._ema_speed,
            'q_scale': q_scale,
            'innovation': innovation.copy()
        }
        return self.x[:2].copy(), info

    def reset(self):
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100
        self.initialized = False
        self._ema_speed = 0.0
        self._prev_measurement = None
        self._current_q_scale = 1.0
