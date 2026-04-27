import torch
import torch.nn as nn
from config import *

class LightweightAttention(nn.Module):
    """轻量级注意力模块，减少计算开销"""
    def __init__(self, hidden_dim, reduction=8):
        super(LightweightAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // reduction, hidden_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        ConvLSTM单元
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # 分别对应遗忘门、输入门、候选细胞状态、输出门
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # 确保hidden_dim和kernel_size是一致的格式
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement state initialization logic
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(cur_layer_input.size(1)):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]

        return layer_output_list, [h, c]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all(isinstance(elem, tuple) for elem in kernel_size))):
            raise ValueError('[kernel_size](file://e:\school\毕设\convlstm-eyetracking\convlstm.py#L0-L0) must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ResidualBlock(nn.Module):
    """残差块，用于在不增加太多参数的情况下提升网络深度"""
    def __init__(self, channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                               stride=stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        out = self.relu(out)
        return out


class PupilTrackingConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=(3, 3), num_layers=2, dropout_rate=0.5):
        """
        基于分割的瞳孔追踪模型 - 使用ConvLSTM进行时序分割和回归
        """
        super(PupilTrackingConvLSTM, self).__init__()
        
        # 使用残差块改进的空间特征提取器
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            ResidualBlock(32),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            ResidualBlock(64),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # ConvLSTM层 - 保持原有参数
        self.convlstm = ConvLSTM(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 轻量级注意力机制
        self.attention = LightweightAttention(hidden_dim, reduction=8)
        
        # 简化但有效的分割解码器
        self.segmentation_decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 1, kernel_size=1),  # 输出单通道分割掩码
            nn.Sigmoid()  # 输出0-1之间的分割概率
        )
        
        # 改进的回归分支 - 使用轻量级注意力
        self.regression_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim * (IMG_HEIGHT//4) * (IMG_WIDTH//4), 256),  # 根据实际尺寸调整
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2)
        )
        
        # 时序聚合模块 - 保持简单有效
        self.temporal_aggregator = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_segmentation=False):
        """
        前向传播
        x: (batch_size, seq_len, channels, height, width) - 输入视频序列
        return_segmentation: 是否返回分割掩码
        返回: (batch_size, 2) - 瞳孔中心坐标 (x, y)，可选择性返回分割掩码
        """
        batch_size, seq_len, channels, height, width = x.size()
        
        # 提取每帧的空间特征
        features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (batch, channels, height, width)
            feat = self.spatial_extractor(frame)  # (batch, hidden_dim, h//4, w//4)
            features.append(feat)
        
        # 组合成序列
        features = torch.stack(features, dim=1)  # (batch, seq_len, hidden_dim, h//4, w//4)
        
        # 通过ConvLSTM进行时序建模
        lstm_features, _ = self.convlstm(features)
        lstm_features = lstm_features[0]  # (batch, seq_len, hidden_dim, h//4, w//4)
        
        # 应用轻量级注意力机制
        attended_lstm_features = []
        for t in range(seq_len):
            feat_t = lstm_features[:, t, :, :, :]  # (batch, hidden_dim, h//4, w//4)
            attended_feat = self.attention(feat_t)  # (batch, hidden_dim, h//4, w//4)
            attended_lstm_features.append(attended_feat)
        
        attended_lstm_features = torch.stack(attended_lstm_features, dim=1)  # (batch, seq_len, hidden_dim, h//4, w//4)
        
        # 回归分支 - 使用注意力加权后的特征
        # 平均所有时间步的加权特征
        avg_attended_features = torch.mean(attended_lstm_features, dim=1)  # (batch, hidden_dim, h//4, w//4)
        regression_output = self.regression_branch(avg_attended_features)
        
        # 对每帧生成分割掩码（使用注意力处理后的特征）
        segmentation_masks = []
        for t in range(seq_len):
            mask = self.segmentation_decoder(attended_lstm_features[:, t, :, :, :])  # (batch, 1, height, width)
            segmentation_masks.append(mask)
        
        # 组合所有分割掩码
        segmentation_masks = torch.stack(segmentation_masks, dim=2)  # (batch, 1, seq_len, height, width)
        
        # 时序聚合 - 融合多帧分割结果
        aggregated_mask = self.temporal_aggregator(segmentation_masks)  # (batch, 1, seq_len, height, width)
        
        # 取平均得到最终分割掩码
        final_mask = torch.mean(aggregated_mask, dim=2, keepdim=False)  # (batch, 1, height, width)
        
        # 计算分割掩码的质心作为瞳孔中心
        center_coords = self.compute_centroid_from_mask(final_mask, height, width)
        
        if return_segmentation:
            return center_coords, final_mask, torch.sigmoid(regression_output)
        else:
            # 使用动态融合策略
            # 计算分割掩码的置信度（掩码值的总和）
            seg_confidence = torch.sum(final_mask.view(batch_size, -1), dim=1, keepdim=True)
            seg_confidence = torch.sigmoid(seg_confidence / 100)  # 归一化到0-1
            
            # 动态融合两个分支的结果
            dynamic_weight = seg_confidence.unsqueeze(1)  # 使用分割置信度作为权重
            fused_coords = dynamic_weight * center_coords + (1 - dynamic_weight) * torch.sigmoid(regression_output)
            
            return fused_coords
    
    def compute_centroid_from_mask(self, mask, height, width):
        """
        从分割掩码计算质心坐标
        mask: (batch_size, 1, height, width) - 分割掩码
        返回: (batch_size, 2) - 质心坐标 (x, y)
        """
        batch_size = mask.size(0)
        
        # 获取当前掩码的实际高度和宽度
        actual_height, actual_width = mask.shape[2], mask.shape[3]
        
        # 创建坐标网格
        y_coords = torch.arange(actual_height, dtype=torch.float32, device=mask.device).view(1, 1, -1, 1).repeat(batch_size, 1, 1, actual_width)
        x_coords = torch.arange(actual_width, dtype=torch.float32, device=mask.device).view(1, 1, 1, -1).repeat(batch_size, 1, actual_height, 1)
        
        # 计算加权坐标（使用掩码作为权重）
        mask_flat = mask.view(batch_size, -1)  # (batch, height*width)
        x_coords_flat = x_coords.view(batch_size, -1)  # (batch, height*width)
        y_coords_flat = y_coords.view(batch_size, -1)  # (batch, height*width)
        
        # 确保三个flat tensor具有相同的最后一个维度
        assert mask_flat.shape[1] == x_coords_flat.shape[1] == y_coords_flat.shape[1], \
            f"Mismatched dimensions: mask_flat={mask_flat.shape[1]}, x_coords_flat={x_coords_flat.shape[1]}, y_coords_flat={y_coords_flat.shape[1]}"
        
        # 计算质心
        sum_mask = torch.sum(mask_flat, dim=1, keepdim=True)  # (batch, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-8)  # 防止除零
        
        centroid_x = torch.sum(x_coords_flat * mask_flat, dim=1, keepdim=True) / sum_mask  # (batch, 1)
        centroid_y = torch.sum(y_coords_flat * mask_flat, dim=1, keepdim=True) / sum_mask  # (batch, 1)
        
        # 归一化到[0,1]范围
        norm_x = centroid_x / actual_width
        norm_y = centroid_y / actual_height
        
        # 拼接坐标
        centroids = torch.cat([norm_x, norm_y], dim=1)  # (batch, 2)
        
        return centroids