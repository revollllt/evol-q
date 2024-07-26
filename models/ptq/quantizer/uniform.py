# Adapted from MEGVII Inc.
# https://github.com/megvii-research/FQ-ViT/tree/main

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseQuantizer

class STEFunction(torch.autograd.Function):        # @ Victor: Straight-Through Estimator (STE)，主要用于在量化过程中保留梯度信息
    def forward(ctx, input):                       # @ Victor: STE的基本思想是在正向传播中使用离散的量化值，而在反向传播中使用原始的浮点数值来计算梯度
        return (input > 0).float()

    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)
    
def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return x + (x.round() - x).detach()

class UniformQuantizer(BaseQuantizer):                                            # @ Victor: 标准量化（见 FQ-ViT 里面的解释）

    def __init__(self, bit_type, observer, module_type, bcorr_weights):
        super(UniformQuantizer, self).__init__(bit_type, observer, module_type, bcorr_weights)
        self.scale = None
        self.zero_point = None

    def update_quantization_params(self, *args, **kwargs):                        # @ Victor: 更新量化参数
        self.scale, self.zero_point = self.observer.get_quantization_params(
            *args, **kwargs)
        self.scale = torch.nn.Parameter(self.scale, requires_grad=True)

    def quant(self, inputs, scale=None, zero_point=None):                         # @ Victor: 量化处理
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        scale = scale.to(inputs.device)
        zero_point = zero_point.to(inputs.device)
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        # self.thd_pos = 2**8 - 1
        # s_grad_scale = 1.0 / ((self.thd_pos * inputs.numel()) ** 0.5)
        # scale = grad_scale(scale, s_grad_scale)
        zero_point = zero_point.reshape(range_shape)
        outputs = inputs / scale + zero_point
        # round_ste = STEFunction()
        outputs = round_ste(outputs).clamp(self.bit_type.lower_bound,
                                        self.bit_type.upper_bound)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):                    # @ Victor: 还原浮点数处理
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        scale = scale.to(inputs.device)
        zero_point = zero_point.to(inputs.device)
        range_shape = self.get_reshape_range(inputs)
        scale = scale.reshape(range_shape)
        zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs
