
import torch
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear

class BaseQuant(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def model_quant(self):                                                 # @ Victor: 其实就是把所有 Q 字头的层的 "quant" 参数打开
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    m.mode = 'int'

    def model_dequant(self):                                               # @ Victor: 其实就是把所有 Q 字头的层的 "quant" 参数关闭
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

