import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
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
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
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
            raise ValueError('kernel_size must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class PupilTrackingConvLSTM(nn.Module):
    """瞳孔追踪模型：CNN编码 + ConvLSTM时序 + 热力图质心定位"""
    def __init__(self, input_dim=1, hidden_dim=48, kernel_size=(3, 3), num_layers=1):
        super(PupilTrackingConvLSTM, self).__init__()

        # 空间特征提取器：1→16→32→hidden_dim，仅一次下采样（22×30 热力图）
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),                        # (16, H, W)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # (32, H/2, W/2)

            nn.Conv2d(32, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15)                          # (hidden_dim, H/2, W/2)
        )

        # ConvLSTM 时序建模
        self.convlstm = ConvLSTM(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 热力图解码器：hidden_dim → 1，上采样回原图尺寸
        self.heatmap_head = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(hidden_dim, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # 上采样倍率：一次 MaxPool 后尺寸为 H/2, W/2，放大 2 倍回原图
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, return_segmentation=False):
        """
        x: (batch, seq_len, channels, height, width)
        返回: (batch, 2) 归一化瞳孔坐标 [x, y]
        """
        _, seq_len, _, _, _ = x.size()

        # 逐帧提取空间特征
        features = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]
            feat = self.spatial_extractor(frame)          # (batch, hidden_dim, H/2, W/2)
            features.append(feat)

        features = torch.stack(features, dim=1)            # (batch, seq_len, hidden_dim, H/2, W/2)

        # ConvLSTM 时序建模 — 最后一个时间步的输出已聚合全序列信息
        lstm_out, _ = self.convlstm(features)
        lstm_out = lstm_out[0]                             # (batch, seq_len, hidden_dim, H/2, W/2)
        last_feat = lstm_out[:, -1, :, :, :]               # (batch, hidden_dim, H/2, W/2)

        # 生成热力图 → 上采样 → 计算质心
        heatmap = self.heatmap_head(last_feat)             # (batch, 1, H/2, W/2)
        heatmap = self.upsample(heatmap)                   # (batch, 1, H, W)
        heatmap = torch.sigmoid(heatmap)

        # 从热力图加权平均计算瞳孔坐标（保留空间信息，实现亚像素精度）
        coords = self._compute_centroid(heatmap)

        if return_segmentation:
            return coords, heatmap, coords
        return coords

    def _compute_centroid(self, heatmap):
        """
        从热力图计算质心坐标。
        heatmap: (batch, 1, H, W)
        返回: (batch, 2) 归一化坐标 [x, y]
        """
        batch_size, _, h, w = heatmap.shape

        y_coords = torch.arange(h, dtype=torch.float32, device=heatmap.device).view(1, 1, -1, 1).repeat(batch_size, 1, 1, w)
        x_coords = torch.arange(w, dtype=torch.float32, device=heatmap.device).view(1, 1, 1, -1).repeat(batch_size, 1, h, 1)

        mask_flat = heatmap.view(batch_size, -1)
        x_coords_flat = x_coords.view(batch_size, -1)
        y_coords_flat = y_coords.view(batch_size, -1)

        sum_mask = torch.clamp(torch.sum(mask_flat, dim=1, keepdim=True), min=1e-8)

        centroid_x = torch.sum(x_coords_flat * mask_flat, dim=1, keepdim=True) / sum_mask
        centroid_y = torch.sum(y_coords_flat * mask_flat, dim=1, keepdim=True) / sum_mask

        centroids = torch.cat([centroid_x / w, centroid_y / h], dim=1)
        return centroids
