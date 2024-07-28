import os
import shutil


# 定义图片文件夹和目标文件夹
image_folder = '/home/zou/data/ImageNet/ILSVRC2012_img_val'
target_folder = '/home/zou/data/ImageNet/val'

# 创建目标文件夹，如果不存在则创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 读取分类文件
with open('/home/zou/codes/python/val_class.txt', 'r') as file:
    for line in file:
        image_name, class_id = line.strip().split()
        class_id = int(class_id)
        
        # 创建类别文件夹
        class_folder = os.path.join(target_folder, str(class_id))
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        # 移动图片到对应的类别文件夹
        source_path = os.path.join(image_folder, image_name)
        target_path = os.path.join(class_folder, image_name)
        if os.path.exists(source_path):
            shutil.copy(source_path, target_path)
        else:
            print(f"Image {source_path} not found.")
            
print("Images have been sorted into their respective class folders.")