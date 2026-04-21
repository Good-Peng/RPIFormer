import torch
import torch.nn as nn

"""
    通过相位（欧拉公式）提取条形（水平、垂直）、通道特征：
        主要功能：对图像特征进行多方向的特征提取、相位计算、欧拉公式展开、特征处理和融合，以增强特征的表达能力，为后续的图像任务提供更丰富的特征信息。
        实现方式：先计算水平和垂直方向的相位，提取相应方向的振幅，运用欧拉公式展开特征，接着对展开后的特征进行处理，同时提取通道方向的特征，
                最后将原始输入特征和处理后的多方向特征进行拼接并融合，输出处理后的特征张量。
"""
class HeightWidthChannelEulerProcessor(nn.Module):
    def __init__(self, input_dimension, use_bias=False, operation_mode='fc'):  # 构造函数
        super().__init__()  # 初始化父类

        self.channel_normalization = nn.BatchNorm2d(input_dimension)

        self.horizontal_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)  # 用于提取水平方向的特征
        self.vertical_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)  # 用于提取垂直方向的特征

        self.channel_feature_conv = nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=use_bias)  # 用于提取通道方向的特征
        self.horizontal_transform_conv = nn.Conv2d(2 * input_dimension, input_dimension, (1, 5), stride=1,
                                                  padding=(0, 5 // 2), groups=input_dimension, bias=False)  # 用于处理水平方向的特征
        self.vertical_transform_conv = nn.Conv2d(2 * input_dimension, input_dimension, (5, 1), stride=1,
                                                 padding=(5 // 2, 0), groups=input_dimension, bias=False)  # 用于处理垂直方向的特征

        self.operation_mode = operation_mode
        if operation_mode == 'fc':
            self.horizontal_phase_calculator = nn.Sequential(  # 水平方向相位计算
                nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=True),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU()
            )
            self.vertical_phase_calculator = nn.Sequential(  # 垂直方向相位计算
                nn.Conv2d(input_dimension, input_dimension, 1, 1, bias=True),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU()
            )
        else:
            self.horizontal_phase_calculator = nn.Sequential(  # 水平方向相位计算
                nn.Conv2d(input_dimension, input_dimension, 3, stride=1, padding=1, groups=input_dimension, bias=False),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU()
            )
            self.vertical_phase_calculator = nn.Sequential(  # 垂直方向相位计算
                nn.Conv2d(input_dimension, input_dimension, 3, stride=1, padding=1, groups=input_dimension, bias=False),
                nn.BatchNorm2d(input_dimension),
                nn.ReLU()
            )

        self.feature_fusion_layer = nn.Sequential(
            # 添加2D卷积层，输入通道数是原通道数的4倍（因为经过DWT后会产生4个子带），输出通道数为input_dimension
            nn.Conv2d(4 * input_dimension, input_dimension, kernel_size=1, stride=1),
            # 批量归一化层，对input_dimension个特征图进行归一化
            nn.BatchNorm2d(input_dimension),
            # ReLU激活函数，inplace=True表示直接在输入数据上进行修改以节省内存
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor):  # 前向传播函数
        horizontal_phase = self.horizontal_phase_calculator(input_tensor)  # 计算水平方向的相位
        vertical_phase = self.vertical_phase_calculator(input_tensor)  # 计算垂直方向的相位

        horizontal_amplitude = self.horizontal_feature_conv(input_tensor)  # 提取水平方向的振幅
        vertical_amplitude = self.vertical_feature_conv(input_tensor)  # 提取垂直方向的振幅

        # 欧拉公式的特征加权/融合
        horizontal_euler = torch.cat([horizontal_amplitude * torch.cos(horizontal_phase),
                                      horizontal_amplitude * torch.sin(horizontal_phase)], dim=1)  # 欧拉公式展开水平方向
        vertical_euler = torch.cat([vertical_amplitude * torch.cos(vertical_phase),
                                    vertical_amplitude * torch.sin(vertical_phase)], dim=1)  # 欧拉公式展开垂直方向

        original_input = input_tensor
        transformed_h = self.horizontal_transform_conv(horizontal_euler)  # 处理水平方向的特征 H torch.Size([10, 64, 32, 32])
        transformed_w = self.vertical_transform_conv(vertical_euler)  # 处理垂直方向的特征 W
        transformed_c = self.channel_feature_conv(input_tensor)  # 提取通道方向的特征 C

        merged_features = torch.cat([original_input, transformed_h, transformed_w, transformed_c], dim=1)  # torch.Size([50, 9, 224, 224])


        output_tensor = self.feature_fusion_layer(merged_features)
        return output_tensor  # 返回处理后的张量

if __name__ == '__main__':
    euler_processor = HeightWidthChannelEulerProcessor(input_dimension=64)
    input_data = torch.rand(1, 64, 50, 50)
    output_data = euler_processor(input_data)
    print(f'Input size: {input_data.size()}')
    print(f'Output size: {output_data.size()}')


