import argparse
import os
from utils import *
from models import *
from joint_evol_opt import JointQuantization

# @Zou ----------------------------------------------------------------------------------#
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    RealLabelsImagenet,
)
# @Zou ----------------------------------------------------------------------------------#

from hooks.hooks import *

parser = argparse.ArgumentParser(description='CPT-V')

parser.add_argument('model',                                                                                                 # @ Victor: 似乎可以自己加入新模型
                    choices=[                                                                                                # @ Victor: 这里的选项会被 str2model(name) 处理
                        'deit_tiny', 'deit_small', 'deit_base', 'vit_base',
                        'vit_large', 'swin_tiny', 'swin_small', 'swin_base',
                        'levit_128s', 'levit_128', 'levit_192', 'levit_256', 'levit_384',
                        'fastvit_sa12', 'fastvit_t8', 'fastvit_s12'
                    ],
                    help='model')
parser.add_argument('data', metavar='DIR', help="ImageNet file path")
parser.add_argument('--save_folder', default=False, help='path for storing checkpoints and results')
parser.add_argument('--ptf', default=False, action='store_true', help="power of two activation quantization")
parser.add_argument('--lis', default=False, action='store_true', help="log-int-softmax from FQ-ViT. Not used in CPT-V initialization due to poor performance")
parser.add_argument('--bias-corr', default=False, action='store_true')
parser.add_argument('--mode', default="layerwise", choices=["fp_no_quant", "fq_vit", "fq++", "evolq", "e2e"])               # @ Victor: 模式选择
parser.add_argument('--quant-method', default='minmax', choices=['minmax', 'ema', 'omse', 'percentile'], help="quantization scheme for initialized model")
parser.add_argument('--w_bit_type', default='int8', choices=['int3', 'uint3', 'uint4', 'uint8', 'int4', 'int8', 'fp32',])   # @ Victor: W
parser.add_argument('--a_bit_type', default='uint8', choices=['uint4', 'uint8', 'int4', 'int8', 'fp32',])                   # @ Victor: A
parser.add_argument('--calib-batchsize', default=100, type=int, help='batchsize of calibration set')
parser.add_argument('--calib-size', default=1000, type=int, help="size of calibration dataset")
parser.add_argument('--val-batchsize', default=8, type=int, help='batchsize of validation set')
parser.add_argument('--num-workers',
                    default=16,
                    type=int,
                    help='number of data loading workers (default: 16)')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--num_passes', default=10, type=int, help="number of passes across all blocks (P)")
parser.add_argument('--num_cycles', default=3, type=int, help="number of cycles per blocks (C)")    # ! 这里原本代码写错了，number of cycles 在论文中不是 K，而应该是 C。已修改
parser.add_argument('--temp', default=3.0, type=float, help='temperature')
parser.add_argument('--loss', default='contrastive', choices=['contrastive','mse', 'kl', 'cosine'], help="loss function for evolutionary search's fitness function")
parser.add_argument('--img_size', default=224, type=int) 
# parser.add_argument('--envs', default="server", choices=['server', 'mypc'])

def str2model(name):
    d = {
        'deit_tiny'  : deit_tiny_patch16_224,                                      # @ Victor: 在 vit_quant.py 内所定义
        'deit_small' : deit_small_patch16_224,                                     # @ Victor: 在 vit_quant.py 内所定义
        'deit_base'  : deit_base_patch16_224,                                      # @ Victor: 在 vit_quant.py 内所定义
        'vit_base'   : vit_base_patch16_224,                                       # @ Victor: 在 vit_quant.py 内所定义
        'vit_large'  : vit_large_patch16_224,                                      # @ Victor: 在 vit_quant.py 内所定义
        'swin_tiny'  : swin_tiny_patch4_window7_224,                               # @ Victor: 在 swin_quant.py 内所定义
        'swin_small' : swin_small_patch4_window7_224,                              # @ Victor: 在 swin_quant.py 内所定义
        'swin_base'  : swin_base_patch4_window7_224,                               # @ Victor: 在 swin_quant.py 内所定义
        'levit_128s' : levit_128s,                                                 # @ Victor: 在 levit_quant.py 内所定义
        'levit_128'  : levit_128,                                                  # @ Victor: 在 levit_quant.py 内所定义
        'levit_192'  : levit_192,                                                  # @ Victor: 在 levit_quant.py 内所定义
        'levit_256'  : levit_256,                                                  # @ Victor: 在 levit_quant.py 内所定义
        'levit_384'  : levit_384,                                                  # @ Victor: 在 levit_quant.py 内所定义
        'fastvit_sa12' : fastvit_sa12,                                             # @ Zou: 在 fastvit_quant.py 内所定义
        'fastvit_t8' : fastvit_t8,
        'fastvit_s12' : fastvit_s12,
    }
    print('Model: %s' % d[name].__name__)
    return d[name]


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)                                                  # @ Victor: 设置递归调用的最大深度
    os.environ['PYTHONHASHSEED'] = str(seed)                                       # @ Victor: 设置环境变量
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'                              # @ Victor: 设置环境变量
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parser.parse_args()
    seed(args.seed)

    device = torch.device(args.device)

    if args.mode == "fq_vit" or args.mode == "e2e":
        from config_fq import Config
    else:
        from config import Config

    cfg = Config(args)
    model_quant = str2model(args.model)(pretrained=True, cfg=cfg)                        # @ Victor: 导入pretrained模型
    model_quant = model_quant.to(device)                                                       # @ Victor: 将模块及其所有子模块的参数和缓冲区移动到指定的设备上（PyTorch）

    model_without_quant = str2model(args.model)(pretrained=True, cfg=cfg)
    model_without_quant = model_without_quant.to(device)
    
    
    # model.stem.register_forward_hook(stem_hook_quant.hook_fn)
    # model_without_quant.stem.register_forward_hook(stem_hook_without_quant.hook_fn)
    
    
    dataset = create_dataset(                                                       # @ Zou: new val datasetm, the same as fastvit ml-fastvit/validate.py
        root=args.data,
        name="",
        )
    val_loader = create_loader(
        dataset,
        input_size=[3, args.img_size, args.img_size],
        batch_size=args.val_batchsize,
        use_prefetcher=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        crop_pct=0.875,
        pin_memory=True
    )
    # switch to evaluate mode
    model_quant.eval()                                                                   # @ Victor: 切换到 evaluate 模式

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    if not args.mode == "fp_no_quant": #check if in a quantization mode            # @ Victor: 如果用 "fp_no_quant" 那就是非量化模式，否则就会进行量化
        # train_dataset = datasets.ImageFolder(traindir, train_transform)
        _, calib_dataset = torch.utils.data.random_split(dataset, [len(dataset)-args.calib_size, args.calib_size])  # @ Victor: 从训练集中随机划分出一部分作为校准数据集
        
        calib_loader = create_loader(                                              # @ Zou: new calib_loader
            calib_dataset,
            input_size=[3, args.img_size, args.img_size],
            batch_size=args.calib_batchsize,
            use_prefetcher=True,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            crop_pct=0.875,
            pin_memory=True
        )

        if args.mode == "fq++" or args.mode == "fq_vit" or args.mode == "e2e":
            
            model_quant.model_open_calibrate()                                           # @ Victor: 开启模型的校准模式
            with torch.no_grad():                                                  # @ Victor: 无梯度计算校准。（禁用梯度计算，加速推理过程，节省显存）

                for i, (image, target) in enumerate(calib_loader):                 # @ Victor: 遍历 calib_loader，将图像数据移动到指定设备，并输入模型进行前向传播
                    image = image.to(device)
                    if i == len(calib_loader) - 1:                                 # @ Victor: 如果是最后一个批次，则使用 OMSE 方法计算最小量化误差
                        # This is used for OMSE method to
                        # calculate minimum quantization error
                        model_quant.model_open_last_calibrate()                          # @ Victor: 函数定义见 base_quant.py 和 swin_quant.py
                    model_quant(image)
            model_quant.model_close_calibrate()                                          # @ Victor: 关闭模型的校准模式
            
            # 测试单个module的量化
            # quant_stem(model_quant)                                                     # @ Zou: 函数定义见 hooks/hooks.py
            # print_quantization_params(model_quant, args.save_folder+'/sa12_quantization_params.txt')                                       # @ Zou: 函数定义见 hooks/hooks.py
            

            print('Validating layerwise quantization...')
            val_loss, val_prec1, val_prec5 = validate_with_hook(args, val_loader, model_quant, model_without_quant,     # @ Victor: 验证量化后的模型  # @ Zou: fastvit此处会出现time变大问题导致out of memory
                                                    criterion, device)

                
    #     if args.mode == "evolq" or args.mode == "e2e":

    #         print("Loading Model...")
    #         model = torch.load(args.save_folder+"/model_layerwise.pt").to("cpu")   # MARK: 加载已经逐层量化和校准后的模型作为基础
    #         optim = JointQuantization(model, calib_loader, device, args, val_loader=val_loader)    # @ Victor: 联合量化见 joint_evol_opt.py
    #         model = optim.opt()                                                                    # @ Victor: 实际运行联合量化

    #         print('Validating Evol-Q optimization...')
    #         model.model_quant()                                                    # @ Victor: 打开量化开关。函数定义见 base_quant.py 和 swin_quant.py # @ Zou: new calib dataset
    #         val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,     # @ Victor: 验证量化后的模型
    #                                                 criterion, device)
    #         with open(args.save_folder+"/evolq.txt", "w") as f:
    #             f.write(str(val_prec1)+"\n")
    # else:

    #     print('Validating full precision...')
    #     val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,         # @ Victor: 验证全精度的模型
    #                                               criterion, device)

if __name__ == '__main__':
    main()
