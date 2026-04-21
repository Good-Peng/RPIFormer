from .losses import (L1Loss, MSELoss, PSNRLoss, CharbonnierLoss)
"""
    L1Loss：L1 损失函数，计算预测值和真实值之间的绝对误差
    MSELoss：均方误差损失函数，计算误差的平方平均值
    PSNRLoss：峰值信噪比损失，常用于图像重建质量评估
    CharbonnierLoss：Charbonnier 损失，L1 的平滑版本，更稳健
"""
# 显示暴露接口 外部使用
__all__ = [
    'L1Loss', 'MSELoss', 'PSNRLoss', 'CharbonnierLoss',
]
