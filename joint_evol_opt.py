
from models.levit_quant import Attention, AttentionSubsample
from models.vit_quant import Attention_ViT
from models.swin_quant import WindowAttention
import numpy as np
from models import *
import torch
from utils import *
import heapq
import random
import time

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
 
class JointQuantization:
    def __init__(self, inf_model, calib_loader, device, args, val_loader=None):
        self.device = device
        inf_model = inf_model.to(device)                                         # @ Victor: 应该是指 inference model (模型在传入的时候已经是 model.eval()了的)
        self.inf_model = inf_model
        self.calib_loader = calib_loader
        self.calib_samples = len(calib_loader)
        self.criterion = torch.nn.CrossEntropyLoss() #.to(device)
        self.args = args
        if val_loader is not None:
            self.val_loader = val_loader
        self.inf_model.model_quant()
        self.inf_model.eval()


        self.mods_to_optimize = [QLinear]


    def loss(self, scales):
        self.set_scales(scales)                                                  # @ Victor: 首先 set_scales
        
        res = 0
        for i, (x, _) in enumerate(self.calib_loader):                           # @ Victor: 从 calib_loader 数据加载器中遍历每个批次的数据
            x = x.cuda()                                                         # @ Victor: 数据 x 被转移到 GPU 上进行计算

            with torch.no_grad():                                                # @ Victor: 在不计算梯度的情况下
                o_quant_x0 = self.inf_model(x)                                   # @ Victor: 通过 inf_model 模型进行前向传播，得到量化后的输出 o_quant_x0
                o_quant_x0 = o_quant_x0.to("cpu")                                # @ Victor: 并将其转移回 CPU 上

                o_fp = self.o_fps[i].to("cpu")                                   # @ Victor: 获取原始模型的输出，并转移到 CPU 上以进行后续计算

                if self.args.loss == "contrastive":                              # @ Victor: 是否采用对比学习损失函数（contrastive losses）
                    loss = contrastive_loss(o_quant_x0, o_fp, self.args.temp)    # @ Victor: 具体见 utils.py（所实现的对比学习损失函数其实就是论文里说的infoNCE）
                elif self.args.loss == "mse":                                    # @ Victor: mean squared error
                    loss = self.mse(o_quant_x0, o_fp)
                elif self.args.loss == "cosine":                                 # @ Victor: cosine similarity
                    loss = torch.sum(self.cos(o_quant_x0, o_fp))
                elif self.args.loss == "kl":                                     # @ Victor: KL divergence
                    loss = self.kl(o_quant_x0, o_fp)
                res += float(loss.item())
                
                torch.cuda.empty_cache()
        res = res / len(self.calib_loader)                                       # @ Victor: 将总损失值 res 除以数据加载器中的批次数量，得到平均损失
        return res

    def set_scales(self, scales):                                                # @ Victor: 用于将一个一维数组中的 scale 值按模块位置重新设置到模型的对应模块中
        ind = 0
        for mod in self.m.modules():
            if type(mod) in self.mods_to_optimize:
                current_scale = scales[self.m.indices[ind]:self.m.indices[ind+1]]     # @ Victor: 根据预先存储的索引 self.m.indices，提取当前模块对应的 scale 值
                current_scale = current_scale.reshape(mod.quantizer.scale.shape)      # @ Victor: 将提取的 scale 值调整为当前模块量化器 scale 的形状
                mod.quantizer.scale = torch.nn.Parameter(torch.Tensor(current_scale).to(mod.quantizer.scale.device))    # @ Victor: 将调整后的 scale 值转换为 PyTorch 的参数，并设置到当前模块的量化器中，确保它在正确的设备上
                ind += 1

    def get_scales(self):                          # @ Victor: 用于从模型中提取所有需要优化的模块的 scale 值，并将它们存储为一个一维数组，同时记录每个模块 scale 值在数组中的位置
        scales = []

        index = 0
        self.m.indices = [index]

        for mod in self.m.modules():
            if type(mod) in self.mods_to_optimize:
                scale = mod.quantizer.scale
                scale = scale.cpu().detach().numpy().flatten()                       # @ Victor: 将 scale 值从设备转移到 CPU，分离梯度，转换为 numpy 数组，并展开为一维
                scales.append(scale)
                index += scale.size
                self.m.indices.append(index)

        return np.concatenate(scales)

    def mutate(self, scales):                                                         # @ Victor: 【进化算法：变异】
        if self.bits == 4 or self.bits == 3:                                          # @ Victor: 设置随机扰动范围 (ϵ controls the size of the uniform ball)
            rng = 1e-3                                                                # @ Victor: ϵ = 10^-3 for 4W8A and 3W8A
        elif self.bits == 8:                                                          # @ Victor: 设置随机扰动范围 (for 8 bits)
            rng = 1e-4                                                                # @ Victor: ϵ = 10^-4 for 8W8A
        else:
            print("Range is not tested for # bits != 3, 4 or 8.")
            rng=1e-3

        perturbations = np.random.uniform(low=-1*rng, high=rng, size=scales.shape)    # @ Victor: 生成与 scales 形状相同的随机扰动 perturbations，其值在-rng到rng之间均匀分布
        return np.abs(scales + perturbations)                                         # @ Victor: 将这些扰动加到原始的 scales 上，并取绝对值返回

    def evolutionary_algorithm(self, scales, num_cycles):                             # @ Victor: 【进化算法：主体，Block-wise Evolutionary Search】

        pop_size = 15                                                                 # @ Victor: population size (K=15)
        population = []                                                               # @ Victor: 用于存储种群中的个体

        # make best layer-wise scales into initial population
        obj = self.loss(scales)
        for p in range(0, pop_size):
            population.append((obj, scales))
        num_samples = 10                                                              # @ Victor: 表示每轮中从种群中抽取的样本数 (S=10)
        best_prev = obj                                                               # @ Victor: 用于存储前一轮中的最佳损失值
        for i in range(0, num_cycles):                                                # @ Victor: 在论文中 num_cycles 取值为 3（C=3）
            # get sampling of population
            samples = random.choices(population, k=num_samples)

            # get sample with smallest loss
            heapq.heapify(samples)                                                    # @ Victor: 将抽取的样本转换为一个堆
            parent = samples[0]                                                       # @ Victor: 堆顶即为损失值最小的个体

            mutated_scales = self.mutate(parent[1])                                   # @ Victor: 调用 mutate 方法对 parent 的 scales 进行突变，生成 mutated_scales

            tic = time.perf_counter()                                                 # @ Victor: 计算时间（开始）
            obj = self.loss(mutated_scales)                                           # @ Victor: 计算突变后的 mutated_scales 的损失值 
            toc = time.perf_counter()                                                 # @ Victor: 计算时间（结束）
            population.append( (obj, mutated_scales) )                                # @ Victor: 将新的个体 (obj 和 mutated_scales) 添加到种群中
            population = sorted(population, key=lambda t: t[0])                       # @ Victor: 并根据损失值对种群进行排序，保留损失值最小的 pop_size 个个体。
            population.pop()

        heapq.heapify(population)                                                     # @ Victor: 将种群转换为堆
        return heapq.heappop(population)                                              # @ Victor: 并返回堆顶的个体 (即损失值最小的个体)

    def opt(self):                                                                    # @ Victor: 【优化主函数】
        
        attn_modules = []
        for m in self.inf_model.modules():
            if type(m) in [Attention, AttentionSubsample, Attention_ViT]:             # @ Victor: 来自于 vit_quant 和 levit_quant
                attn_modules.append(m)
                m.inputs = []
                m.outputs = []
                m.handle.remove()

            if type(m) in [WindowAttention]:                                          # @ Victor: 来自于 swin_quant
                attn_modules.append(m)
        
        self.o_fps = []
        self.inf_model.model_dequant()                                                # @ Victor: 首先关闭量化开关（获取浮点条件下的输出结果）
        for i, (x, target) in enumerate(self.calib_loader):                           # @ Victor: 遍历数据加载器
            x = x.cuda()
            with torch.no_grad():                                                     # @ Victor: 在禁用梯度计算的上下文中
                o_fp = self.inf_model(x)                                              # @ Victor: 计算模型的输出 o_fp

            self.o_fps.append(o_fp.cpu())                                             # @ Victor: 并将输出存储在 self.o_fps 列表中
        self.inf_model.model_quant()                                                  # @ Victor: 打开量化开关（开始进入Evol-Q的搜索）

        tic = time.perf_counter()                                                     # @ Victor: 计时开始
        print("beginning Evol-Q..")

        attn_modules.reverse()
        for i in range(0, self.args.num_passes):                                      # @ Victor: Number of passes (P=10)
            obj = ""
            j = 0
            for m in attn_modules:                                                    # @ Victor: 所以其实每一个 attn_modules 都会经过 P=10 次的评估
                self.m = m
                j += 1
                init = self.get_scales()

                m_name = m.__class__.__name__
                print("module: ", m_name, str(j))

                if type(m) in [Attention_ViT, WindowAttention]:
                    self.bits = m.proj.quantizer.bit_type.bits
                elif type(m) in [Attention, AttentionSubsample]:
                    self.bits = m.proj[1].quantizer.bit_type.bits

                final_obj, final_scales = self.evolutionary_algorithm(init, self.args.num_cycles)    # @ Victor: 进化算法
                self.set_scales(final_scales)                                                        # @ Victor: 将进化算法得到最终的量化比例 scale 套用上

                self.inf_model.model_quant()
                loss, top1, top5 = validate(self.args, self.val_loader, self.inf_model, self.criterion, self.device)    # @ Victor: 评估模型的性能
                if i == 0 or top1 > current_top1:
                    current_top1 = top1
                    with open(self.args.save_folder+"/" + self.args.mode +".txt", "w") as f:
                            f.write("Current Top1: " + str(top1) + "\n")
                    torch.save(self.inf_model, self.args.save_folder+ "/evolq.pt")            # @ Victor: 保存整个model的状态（包括模型权重参数和模型结构）

        toc = time.perf_counter()                                                             # @ Victor: 计时结束
        print(f"==== FULL EVOL Q Completed in {toc - tic:0.4f} seconds", "== ", str(i))       # @ Victor: 统计时长
        self.inf_model =  torch.load(self.args.save_folder+"/evolq.pt")
        return self.inf_model
