import numpy as np
import cv2

class PupilKalmanFilter:
    def __init__(self, dt=1, process_noise=1e-2, measurement_noise=1e-1):
        """
        初始化卡尔曼滤波器用于瞳孔追踪
        
        参数:
        - dt: 时间步长 (默认为1，因为视频帧是连续的)
        - process_noise: 过程噪声 (系统模型不确定性)
        - measurement_noise: 测量噪声 (观测不确定性)
        """
        self.dt = dt
        
        # 状态向量 [x, y, vx, vy] - 位置和速度
        self.state = np.array([[0.0], [0.0], [0.0], [0.0]])  # [x, y, vx, vy]
        
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # 控制矩阵 (这里没有控制输入，所以为0)
        self.B = np.zeros((4, 1))
        
        # 观测矩阵 - 我们只能观测位置，不能直接观测速度
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # 过程噪声协方差矩阵
        self.Q = np.array([
            [dt**4/4, 0,       dt**3/2, 0      ],
            [0,       dt**4/4, 0,       dt**3/2],
            [dt**3/2, 0,       dt**2,   0      ],
            [0,       dt**3/2, 0,       dt**2  ]
        ]) * process_noise
        
        # 测量噪声协方差矩阵
        self.R = np.eye(2) * measurement_noise  # 2x2 因为我们观测的是 [x, y]
        
        # 误差协方差矩阵
        self.P = np.eye(4) * 1000  # 初始化为较大的不确定性

    def predict(self):
        """预测步骤：基于系统模型预测下一状态"""
        # X(k|k-1) = F * X(k-1|k-1) + B * U(k)
        self.state = np.dot(self.F, self.state)
        
        # P(k|k-1) = F * P(k-1|k-1) * F.T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.state[:2].flatten()  # 返回预测的位置 [x, y]

    def update(self, measurement):
        """更新步骤：使用测量值更新状态估计"""
        # 计算卡尔曼增益
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # 更新状态估计
        y = np.array(measurement).reshape(-1, 1) - np.dot(self.H, self.state)  # 测量残差
        self.state = self.state + np.dot(K, y)
        
        # 更新误差协方差矩阵
        I = np.eye(len(self.state))
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        
        return self.state[:2].flatten()  # 返回更新后的位置 [x, y]
    
    def initialize(self, initial_position):
        """初始化滤波器状态"""
        self.state[0:2] = np.array(initial_position).reshape(2, 1)
        # 速度初始为0
        self.state[2:4] = np.array([0.0, 0.0]).reshape(2, 1)


class AdaptiveKalmanFilter:
    def __init__(self, process_noise_scale=1e-1, measurement_noise_scale=1e-1):
        """
        自适应卡尔曼滤波器，用于瞳孔跟踪
        """
        # 定义状态向量: [x, y, vx, vy] (位置和速度)
        self.kalman = cv2.KalmanFilter(4, 2)  # 4个状态变量，2个测量变量
        
        # 状态转移矩阵 (F) - 匀速模型
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 测量矩阵 (H) - 我们只能观测位置，不能观测速度
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # 过程噪声协方差矩阵 (Q) - 系统模型的不确定性
        # 增加过程噪声，使滤波器更愿意接受新测量
        self.process_noise_scale = process_noise_scale
        self.kalman.processNoiseCov = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) * (process_noise_scale ** 2)
        
        # 测量噪声协方差矩阵 (R) - 观测的不确定性
        # 适当设置测量噪声，不要过大也不要过小
        self.measurement_noise_scale = measurement_noise_scale
        self.kalman.measurementNoiseCov = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.float32) * (measurement_noise_scale ** 2)
        
        # 误差协方差矩阵 (P) - 初始不确定性
        self.kalman.errorCovPost = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32) * 1.0
        
        # 初始化状态
        self.initialized = False
        self.prev_measurement = None
        self.dt = 1  # 时间步长假设为1帧
        
    def update(self, measurement):
        """
        更新卡尔曼滤波器并返回预测结果
        """
        measurement = np.array(measurement, dtype=np.float32).reshape(2, 1)
        
        if not self.initialized:
            # 第一次测量，初始化状态
            self.kalman.statePre = np.array([
                measurement[0, 0],  # x
                measurement[1, 0],  # y
                0,                  # vx
                0                   # vy
            ], dtype=np.float32)
            
            self.kalman.statePost = np.array([
                measurement[0, 0],  # x
                measurement[1, 0],  # y
                0,                  # vx
                0                   # vy
            ], dtype=np.float32)
            
            self.initialized = True
            self.prev_measurement = measurement.copy()
            return measurement.flatten()
        
        # 预测
        prediction = self.kalman.predict()
        
        # 更新
        estimation = self.kalman.correct(measurement)
        
        # 更新之前的测量值
        self.prev_measurement = measurement.copy()
        
        return estimation[:2].flatten()