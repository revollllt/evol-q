# Jumping Through Local Minima: Quantization in the Jagged Loss Landscape of Vision Transformers (Evol-Q) 

**fastvit with evol-q**

## Dataset Preparation
Download the [ImageNet-1K](http://image-net.org/) dataset and structure the data with valprep.py as follows:
```
# you should change the path in valprep.py at first!
python valprep.py
```
dataset tree
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  validation/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```

## Getting Started

To run an initial experiment with 8-bit quantized DeiT-Tiny:

```sh scripts/run.sh```

This script contains details for how to quantize 3,4,8-bit models as well as model architectures.



| Model      | Top-1 Acc(evol-q) | Top-1 Acc(original) |
| ---------- | ----------------- | ------------------- |
| ViT-Base   |                   |                     |
| DeiT-Tiny  |      67.623       |       71.977        |
| DeiT-Small |                   |                     |
| DeiT-Base  |                   |                     |
| LeViT-128S |                   |                     |
| LeViT-128  |                   |                     |
| LeViT-192  |                   |                     |
| LeViT-246  |                   |                     |
| LeViT-384  |                   |                     |
|fastvit-sa12|      81.564       |                     |


