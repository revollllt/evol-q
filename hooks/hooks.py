import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from misc import all_reduce_mean
from utils import AverageMeter, accuracy
from models import *

class Hook:
    def __init__(self):
        self.output = None
        # self.outputs = {}

    def hook_fn(self, module, input, output):
        self.output = output.detach()  # 保存输出特征图
        # key = f"{module.__class__.__name__}_{id(module)}"
        # self.outputs[key] = output.detach()  # @ Zou: 使用字典保存不同模块的输出特征图



def model_quant(model_quant):                                                 # @ Victor: 其实就是把所有 Q 字头的层的 "quant" 参数打开
    for m in model_quant.modules():
        if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
            m.quant = True
        if model_quant.cfg.INT_NORM:
            if type(m) in [QIntLayerNorm]:
                m.mode = 'int'

def quant_stem(model_quant):
    for module in model_quant.patch_embed.modules():
        # if isinstance(module, MobileOneBlock):
        #     module.quant = True
        if type(module) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                module.quant = True
    model_quant.qact_embed.quant = True
    # model_quant.qact_embed.register_forward_hook(hook.hook_fn)
    # for i, module in enumerate(model_quant.patch_embed):
    #     if isinstance(module, MobileOneBlock):
    #         print(f"patch_embed MobileOneBlock {i} quant status: {module.quant}")

    
def quant_network(model_quant, quant_layers=[False, False, False, False, False, False, False, False]):
    for i, module in enumerate(model_quant.network):
        # print(f"network {i} module: {type(module).__name__}")
        # network 0 module: Sequential
        # network 1 module: PatchEmbed
        # network 2 module: Sequential
        # network 3 module: PatchEmbed
        # network 4 module: Sequential
        # network 5 module: PatchEmbed
        # network 6 module: RepCPE
        # network 7 module: Sequential
        if quant_layers[i]:
            for m in module.modules():
                if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                    m.quant = True
        # model_quant.network.register_forward_hook(hook.hook_fn)

def quant_conv_exp(model_quant):
    for module in model_quant.conv_exp.modules():
        if type(module) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                module.quant = True
    # model_quant.conv_exp.register_forward_hook(hook.hook_fn)

def quant_head(model_quant):
    for module in model_quant.head.modules():
        if type(module) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                module.quant = True
    model_quant.qact_out.quant = True
    # model_quant.qact_out.register_forward_hook(hook.hook_fn)

def hook_stem(model_quant, model_without_quant, hook_quant, hook_without_quant):
    model_quant.qact_embed.register_forward_hook(hook_quant.hook_fn)
    model_without_quant.qact_embed.register_forward_hook(hook_without_quant.hook_fn)
    
def hook_network(model_quant, model_without_quant, hook_quant, hook_without_quant):  # @ Zou: 待更新hook nn.ModuleList中的模块
    model_quant.network.register_forward_hook(hook_quant.hook_fn)
    model_without_quant.network.register_forward_hook(hook_without_quant.hook_fn)
    
def hook_conv_exp(model_quant, model_without_quant, hook_quant, hook_without_quant):
    model_quant.conv_exp.register_forward_hook(hook_quant.hook_fn)
    model_without_quant.conv_exp.register_forward_hook(hook_without_quant.hook_fn)
    
def hook_head(model_quant, model_without_quant, hook_quant, hook_without_quant):
    model_quant.qact_out.register_forward_hook(hook_quant.hook_fn)
    model_without_quant.qact_out.register_forward_hook(hook_without_quant.hook_fn)

def validate_with_hook(args, val_loader, model_quant, model_without_quant, criterion, device):
    batch_time = AverageMeter()
    losses_quant = AverageMeter()
    top1_quant = AverageMeter()
    top5_quant = AverageMeter()
    losses_without_quant = AverageMeter()
    top1_without_quant = AverageMeter()
    top5_without_quant = AverageMeter()
    cosine_similarities = AverageMeter()
    top1_errors = AverageMeter()
    top5_errors = AverageMeter()

    # stem_hook_quant = Hook()
    # stem_hook_without_quant = Hook()
    hook_quant = Hook()
    hook_without_quant = Hook()
    # @ Zou: place to hook and quantize modules
    
    quant_network(model_quant, quant_layers=[True, True, True, True, True, True, True, True])
    # switch to evaluate mode
    model_quant.eval()
    model_without_quant.eval()

    val_start_time = end = time.time()
    loop = tqdm(enumerate(val_loader), leave=True, total=len(val_loader))
    for i, (data, target) in loop:
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            
            # model_quant.patch_embed.register_forward_hook(stem_hook_quant.hook_fn)
            # model_without_quant.patch_embed.register_forward_hook(stem_hook_without_quant.hook_fn)
            # hook_network(model_quant, model_without_quant, hook_quant, hook_without_quant)
            model_quant.patch_embed.register_forward_hook(hook_quant.hook_fn)
            model_without_quant.patch_embed.register_forward_hook(hook_without_quant.hook_fn)
            
            output_quant = model_quant(data)
            output_without_quant = model_without_quant(data)
            
            print(hook_quant.output)
            print(hook_without_quant.output)
            # for key, value in hook_quant.outputs.items():
            #     print(f"Module: {key}, Output shape: {value.shape}")
            # for key, value in hook_without_quant.outputs.items():
            #     print(f"Module: {key}, Output shape: {value.shape}")
            
            break
            # stem_feature_map_quant = stem_hook_quant.output
            # stem_feature_map_withou_quant = stem_hook_without_quant.output
            
            # stem_feature_map_quant_flat = stem_feature_map_quant.view(stem_feature_map_quant.size(0), -1)
            # stem_hook_without_quant_flat = stem_feature_map_withou_quant.view(stem_feature_map_withou_quant.size(0), -1)
            
            # cosine_similarity = F.cosine_similarity(stem_feature_map_quant_flat, stem_hook_without_quant_flat, dim=1)
            # data_cpu = data.cpu()
            # imshow(data_cpu)
            
        loss_quant = criterion(output_quant, target)
        loss_without_quant = criterion(output_without_quant, target)

        # measure accuracy and record loss
        prec1_quant, prec5_quant = accuracy(output_quant.data, target, topk=(1, 5))
        prec1_without_quant, prec5_without_quant = accuracy(output_without_quant.data, target, topk=(1, 5))
        # print(f'acc1: {prec1}  acc5: {prec5}  loss: {loss}')
        # break
        losses_quant.update(loss_quant.data.item(), data.size(0))
        top1_quant.update(prec1_quant.data.item(), data.size(0))
        top5_quant.update(prec5_quant.data.item(), data.size(0))
        losses_without_quant.update(loss_without_quant.data.item(), data.size(0))
        top1_without_quant.update(prec1_without_quant.data.item(), data.size(0))
        top5_without_quant.update(prec5_without_quant.data.item(), data.size(0))

        # cosine_similarities.update(cosine_similarity.mean().item(), data.size(0))
        top1_error = prec1_quant.data.item() - prec1_without_quant.data.item()
        top5_error = prec5_quant.data.item() - prec5_without_quant.data.item()
        top1_errors.update(top1_error, data.size(0))
        top5_errors.update(top5_error, data.size(0))
        # print(f'top1: {top1.val:.3f} ({top1.avg:.3f})  top5: {top5.val:.3f} ({top5.avg:.3f}')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loop.set_description('Test ')
        loop.set_postfix_str('Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                            'Prec@1q ({top1q.avg:.3f})   '
                            'Prec@5q ({top5q.avg:.3f})   '
                            'Prec@1 ({top1.avg:.3f})   '
                            'Prec@5 ({top5.avg:.3f})   '
                            'Prec@1_error ({top1_error.avg:.3f})   '
                            'Prec@5_error ({top5_error.avg:.3f})   '.format(
                                batch_time=batch_time,
                                lossq=losses_quant,
                                top1q=top1_quant,
                                top5q=top5_quant,
                                loss=losses_without_quant,
                                top1=top1_without_quant,
                                top5=top5_without_quant,
                                top1_error=top1_errors,
                                top5_error=top5_errors,
                            ))
        # loop.write(f"\ncosine_similarities_stem ({cosine_similarities.avg:.4f})")
    val_end_time = time.time()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
          format(top1=top1_quant, top5=top5_quant, time=val_end_time - val_start_time))

    return all_reduce_mean(losses_quant.avg), all_reduce_mean(top1_quant.avg), all_reduce_mean(top5_quant.avg)