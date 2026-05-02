import torch
import os

# LPW_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "LPW")
# LPW_ROOT = "./LPW"
LPW_ROOT = "/kaggle/input/datasets/wakakaele/eyetracking-lpw/LPW"

RANDOM_SEED = 42  # 随机数种子
k_folds = 3 # K折交叉验证的折数

# 数据增强
DATA_AUGMENTATION = True
AUGMENT_PROBABILITY = 0.7   # 数据增强的概率

# 梯度裁剪
GRAD_CLIP_NORM = 1.0        # 梯度裁剪的最大范数

# 早停
EARLY_STOPPING_PATIENCE= 5         # 允许的无改善epoch数
EARLY_STOPPING_MIN_DELTA= 0.0001    # 判断为改善的最小变化量

# 训练参数
SEQUENCE_LENGTH = 12
IMG_HEIGHT = 60
IMG_WIDTH  = 80
BATCH_SIZE = 64
EPOCHS = 60
LR = 0.0002
WEIGHT_DECAY = 2e-3
HIDDEN_DIM = 64
HEATMAP_LOSS_WEIGHT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型保存路径
# MODEL_SAVE_PATH = "./checkpoints.pth"