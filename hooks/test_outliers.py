import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *

def quant_stem(model_quant, quant=True):
    for i, module in enumerate(model_quant.patch_embed.modules()):
        # if isinstance(module, MobileOneBlock):
        #     module.quant = True
        # print(f"patch_embed {i} module: {type(module).__name__}")
        if type(module) in [QConv2d, QLinear, QAct, QIntSoftmax]:
            # module.quant = quant    # @ Zou: 打开所有stem层的quant参数
            # if i == 4 or i == 6:    # @ Zou: first MobileOneBlock： 3x3 Conv, S=2
            # if i == 10 or i == 12:  # @ Zou: second MobileOneBlock: 3x3 DWConv, S=2
            if i == 16 or i == 18:  # @ Zou: third MobileOneBlock: 1x1 Conv, S=1
                module.quant = quant
            print(f"patch_embed {i} module: {type(module).__name__}, quant status: {module.quant}")
    model_quant.qact_embed.quant = quant