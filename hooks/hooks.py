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
        # self.output = None
        self.outputs = {}

    def hook_fn(self, module, input, output):
        # self.output = output.detach()  # 保存输出特征图
        key = f"{module.__class__.__name__}_{id(module)}" # @ Zou: 添加id防止FastViT.network中有多个相同的模块
        self.outputs[key] = output.detach()  # @ Zou: 使用字典保存不同模块的输出特征图

class Cosine_Similarity_Container:
    def __init__(self):
        self.cosine_similarities = {}
    
    

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

    
def quant_network(model_quant, quant_layers=[False, False, False, False, False, False, False, False]): # @ Zou: len(quant_layers) is 8
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


# @ Zou: 将hook按照FastViT的模块分为stem, network, conv_exp, head四个大部分
def hook_stem(model_quant, model_without_quant): # @ Zou: Create hooks for stem and return them
    hook_quant = Hook()
    hook_without_quant = Hook()
    model_quant.qact_embed.register_forward_hook(hook_quant.hook_fn)
    model_without_quant.qact_embed.register_forward_hook(hook_without_quant.hook_fn)
    return hook_quant, hook_without_quant
    
def hook_network(model_quant, model_without_quant):  # @ Zou: 对FastViT的network部分中ModuleList的8个小部分，每个部分创建一个hook
    hook_quant = Hook()
    hook_without_quant = Hook()
    for i, module in enumerate(model_quant.network):
        model_quant.network[i].register_forward_hook(hook_quant.hook_fn)
        model_without_quant.network[i].register_forward_hook(hook_without_quant.hook_fn)
    return hook_quant, hook_without_quant
    
def hook_conv_exp(model_quant, model_without_quant):
    hook_quant = Hook()
    hook_without_quant = Hook()
    model_quant.conv_exp.register_forward_hook(hook_quant.hook_fn)
    model_without_quant.conv_exp.register_forward_hook(hook_without_quant.hook_fn)
    return hook_quant, hook_without_quant
    
def hook_head(model_quant, model_without_quant):
    hook_quant = Hook()
    hook_without_quant = Hook()
    model_quant.qact_out.register_forward_hook(hook_quant.hook_fn)
    model_without_quant.qact_out.register_forward_hook(hook_without_quant.hook_fn)
    return hook_quant, hook_without_quant

def hook_all(model_quant, model_without_quant):
    stem_hook_quant, stem_hook_without_quant = hook_stem(model_quant, model_without_quant)
    network_hook_quant, network_hook_without_quant = hook_network(model_quant, model_without_quant)
    conv_exp_hook_quant, conv_exp_hook_without_quant = hook_conv_exp(model_quant, model_without_quant)
    head_hook_quant, head_hook_without_quant = hook_head(model_quant, model_without_quant)
    return stem_hook_quant, stem_hook_without_quant, network_hook_quant, network_hook_without_quant, \
        conv_exp_hook_quant, conv_exp_hook_without_quant, head_hook_quant, head_hook_without_quant


def print_hooks_infor(stem_hook_quant=None, stem_hook_without_quant=None, 
                       network_hook_quant=None, network_hook_without_quant=None,
                       conv_exp_hook_quant=None, conv_exp_hook_without_quant=None,
                       head_hook_quant=None, head_hook_without_quant=None):
    if stem_hook_quant:
        for key, value in stem_hook_quant.outputs.items():
            print(f"[stem_hook_quant] Module: {key}, Output shape: {value.shape}")
    if stem_hook_without_quant:
        for key, value in stem_hook_without_quant.outputs.items():
            print(f"[stem_hook_without_quant] Module: {key}, Output shape: {value.shape}")
    if network_hook_quant:
        for i, (key, value) in enumerate(network_hook_quant.outputs.items()):
            print(f"[network_hook_quant_{i}] Module: {key}, Output shape: {value.shape}")
    if network_hook_without_quant:
        for i, (key, value) in enumerate(network_hook_without_quant.outputs.items()):
            print(f"[network_hook_without_quant_{i}] Module: {key}, Output shape: {value.shape}")
    if conv_exp_hook_quant:
        for key, value in conv_exp_hook_quant.outputs.items():
            print(f"[conv_exp_hook_quant] Module: {key}, Output shape: {value.shape}")
    if conv_exp_hook_without_quant:
        for key, value in conv_exp_hook_without_quant.outputs.items():
            print(f"[conv_exp_hook_without_quant] Module: {key}, Output shape: {value.shape}")
    if head_hook_quant:
        for key, value in head_hook_quant.outputs.items():
            print(f"[head_hook_quant] Module: {key}, Output shape: {value.shape}")
    if head_hook_without_quant:
        for key, value in head_hook_without_quant.outputs.items():
            print(f"[head_hook_without_quant] Module: {key}, Output shape: {value.shape}")


def calculate_cosine_similarity(hook_quant, hook_without_quant):
    feature_map_quant=[]
    feature_map_quant_flat=[]
    feature_map_without_quant=[]
    feature_map_without_quant_flat=[]
    
    for i, (key, value) in enumerate(hook_quant.outputs.items()):
        feature_map_quant.append(value)
        feature_map_quant_flat.append(feature_map_quant[i].view(feature_map_quant[i].size(0), -1))
    for i, (key, value) in enumerate(hook_without_quant.outputs.items()):
        feature_map_without_quant.append(value)
        feature_map_without_quant_flat.append(feature_map_without_quant[i].view(feature_map_without_quant[i].size(0), -1))
    
    cosine_similarity = [] if len(feature_map_quant) > 1 else None
    if len(feature_map_quant) > 1:
        for i in range(len(feature_map_quant_flat)):
            cosine_similarity.append(F.cosine_similarity(feature_map_quant_flat[i], feature_map_without_quant_flat[i], dim=1))
    else:
        # print(feature_map_quant_flat[0].shape, feature_map_without_quant_flat[0].shape)
        cosine_similarity = F.cosine_similarity(feature_map_quant_flat[0], feature_map_without_quant_flat[0], dim=1)
        # print(cosine_similarity)
    return cosine_similarity
        

def print_cosine_similarity(cosine_similarities_dict):
    for key, value in cosine_similarities_dict.items():
        if key == "network":
            for i, item in enumerate(value):
                print(f"[{key}_{i}] Cosine Similarity: {item.mean().item()}")
        else:
            print(f"[{key}] Cosine Similarity: {value.mean().item()}")
    


def calculate_cosine_similarity_dict(stem_hook_quant=None, stem_hook_without_quant=None,  
                                network_hook_quant=None, network_hook_without_quant=None,
                                conv_exp_hook_quant=None, conv_exp_hook_without_quant=None,
                                head_hook_quant=None, head_hook_without_quant=None):
    '''
    
    return: a dictionary of cosine similarity of different parts of the model
    '''
    cosine_similarity_dict = {}
    if stem_hook_quant and stem_hook_without_quant:
        cosine_similarity_dict["stem"] = calculate_cosine_similarity(stem_hook_quant, stem_hook_without_quant)
    if network_hook_quant and network_hook_without_quant:
        cosine_similarity_dict["network"] = calculate_cosine_similarity(network_hook_quant, network_hook_without_quant)
    if conv_exp_hook_quant and conv_exp_hook_without_quant:
        cosine_similarity_dict["conv_exp"] = calculate_cosine_similarity(conv_exp_hook_quant, conv_exp_hook_without_quant)
    if head_hook_quant and head_hook_without_quant:
        cosine_similarity_dict["head"] = calculate_cosine_similarity(head_hook_quant, head_hook_without_quant)
    return cosine_similarity_dict


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
            stem_hook_quant, stem_hook_without_quant = hook_stem(model_quant, model_without_quant)
            network_hook_quant, network_hook_without_quant = hook_network(model_quant, model_without_quant)

            output_quant = model_quant(data)
            output_without_quant = model_without_quant(data)
            
            print_hooks_infor(stem_hook_quant=stem_hook_quant, stem_hook_without_quant=stem_hook_without_quant,
                              network_hook_quant=network_hook_quant, network_hook_without_quant=network_hook_without_quant)
            
            # stem_feature_map_quant = stem_hook_quant.output
            # stem_feature_map_withou_quant = stem_hook_without_quant.output
            # stem_feature_map_quant_flat = stem_feature_map_quant.view(stem_feature_map_quant.size(0), -1)
            # stem_hook_without_quant_flat = stem_feature_map_withou_quant.view(stem_feature_map_withou_quant.size(0), -1)
            # cosine_similarity = F.cosine_similarity(stem_feature_map_quant_flat, stem_hook_without_quant_flat, dim=1)
            cosine_similarities_dict = calculate_cosine_similarity_dict(stem_hook_quant=stem_hook_quant, stem_hook_without_quant=stem_hook_without_quant,
                                        network_hook_quant=network_hook_quant, network_hook_without_quant=network_hook_without_quant)

            print_cosine_similarity(cosine_similarities_dict)
            
            break
            
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