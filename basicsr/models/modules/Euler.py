import torch
import torch.nn as nn


class Euler(nn.Module):

    def __init__(self, input_dimension, use_bias=False, operation_mode="fc"):
        super().__init__()
        self.channel_normalization = nn.BatchNorm2d(input_dimension)
        self.horizontal_feature_conv = nn.Conv2d(
            input_dimension, input_dimension, 1, 1, bias=use_bias
        )
        self.vertical_feature_conv = nn.Conv2d(
            input_dimension, input_dimension, 1, 1, bias=use_bias
        )
        self.channel_feature_conv = nn.Conv2d(
            input_dimension, input_dimension, 1, 1, bias=use_bias
        )
        self.horizontal_transform_conv = nn.Conv2d(
            2 * input_dimension,
            input_dimension,
            (1, 5),
            stride=1,
            padding=(0, 5 // 2),
            groups=input_dimension,
            bias=False,
        )
        self.vertical_transform_conv = nn.Conv2d(
            2 * input_dimension,
            input_dimension,
            (5, 1),
            stride=1,
            padding=(5 // 2, 0),
            groups=input_dimension,
            bias=False,
        )
        self.operation_mode = operation_mode
        if operation_mode == "fc":
            self.horizontal_phase_calculator = nn.Sequential(
                nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=True),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU(),
            )
            self.vertical_phase_calculator = nn.Sequential(
                nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=True),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU(),
            )
        else:
            self.horizontal_phase_calculator = nn.Sequential(
                nn.Conv2d(
                    input_dimension,
                    input_dimension,
                    3,
                    stride=1,
                    padding=1,
                    groups=input_dimension,
                    bias=False,
                ),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU(),
            )
            self.vertical_phase_calculator = nn.Sequential(
                nn.Conv2d(
                    input_dimension,
                    input_dimension,
                    3,
                    stride=1,
                    padding=1,
                    groups=input_dimension,
                    bias=False,
                ),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU(),
            )
        self.feature_fusion_layer = nn.Sequential(
            nn.Conv2d(4 * input_dimension, input_dimension, kernel_size=1, stride=1),
            nn.BatchNorm2d(input_dimension),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor):
        horizontal_phase = self.horizontal_phase_calculator(input_tensor)
        vertical_phase = self.vertical_phase_calculator(input_tensor)
        horizontal_amplitude = self.horizontal_feature_conv(input_tensor)
        vertical_amplitude = self.vertical_feature_conv(input_tensor)
        horizontal_euler = torch.cat(
            [
                horizontal_amplitude * torch.cos(horizontal_phase),
                horizontal_amplitude * torch.sin(horizontal_phase),
            ],
            dim=1,
        )
        vertical_euler = torch.cat(
            [
                vertical_amplitude * torch.cos(vertical_phase),
                vertical_amplitude * torch.sin(vertical_phase),
            ],
            dim=1,
        )
        transformed_h = self.horizontal_transform_conv(horizontal_euler)
        transformed_w = self.vertical_transform_conv(vertical_euler)
        transformed_c = self.channel_feature_conv(input_tensor)
        merged_features = torch.cat(
            [input_tensor, transformed_h, transformed_w, transformed_c], dim=1
        )
        return self.feature_fusion_layer(merged_features)
