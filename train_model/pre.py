import glob
import os
from shutil import move
from os import rmdir

target_folder = './tiny-imagenet-200/val/'

# 创建字典保存图片与类别的映射
val_dict = {}
with open('./tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

# 获取所有图片路径
paths = glob.glob('./tiny-imagenet-200/val/images/*')

# 创建对应的分类文件夹
for path in paths:
    file = os.path.basename(path)  # 使用 os.path.basename 来获取文件名
    folder = val_dict.get(file)    # 使用 get 方法避免 KeyError
    if folder:                     # 检查是否找到了类别
        if not os.path.exists(target_folder + str(folder)):
            os.mkdir(target_folder + str(folder))
            os.mkdir(target_folder + str(folder) + '/images')

# 将图片移动到对应文件夹
for path in paths:
    file = os.path.basename(path)  # 使用 os.path.basename 来获取文件名
    folder = val_dict.get(file)    # 使用 get 方法避免 KeyError
    if folder:                     # 检查是否找到了类别
        dest = target_folder + str(folder) + '/images/' + str(file)
        move(path, dest)

# 删除空文件夹
rmdir('./tiny-imagenet-200/val/images')
