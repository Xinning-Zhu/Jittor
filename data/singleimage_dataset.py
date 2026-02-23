import numpy as np
import os.path
import jittor as jt
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random

#用于处理单图像翻译任务。它的核心功能是从两个域（A 和 B）各加载一张图像，并通过复杂的数据增强策略生成大量训练样本，适用于需要从单张图像中学习域转换的生成模型。
class SingleImageDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, 'trainA')  
        self.dir_B = os.path.join(opt.dataroot, 'trainB')  
        # 从两个路径各加载一张图像，并验证是否仅存在一张图像
        if os.path.exists(self.dir_A) and os.path.exists(self.dir_B):
            self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    
        self.A_size = len(self.A_paths)  
        self.B_size = len(self.B_paths)  

        assert len(self.A_paths) == 1 and len(self.B_paths) == 1,\
            "SingleImageDataset class should be used with one image in each domain"
        A_img = Image.open(self.A_paths[0]).convert('RGB')
        B_img = Image.open(self.B_paths[0]).convert('RGB')
        print("Image sizes %s and %s" % (str(A_img.size), str(B_img.size)))

        self.A_img = A_img
        self.B_img = B_img
        
        # 为每个批次生成统一的随机缩放因子，确保同批次缩放一致
        A_zoom = 1 / self.opt.random_scale_max
        zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
        self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, opt.batch_size, 1)), [-1, 2])

        B_zoom = 1 / self.opt.random_scale_max
        zoom_levels_B = np.random.uniform(B_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
        self.zoom_levels_B = np.reshape(np.tile(zoom_levels_B, (1, opt.batch_size, 1)), [-1, 2])

        # 随机打乱补丁的裁剪索引，避免负样本来自相同位置
        self.patch_indices_A = list(range(len(self)))
        random.shuffle(self.patch_indices_A)
        self.patch_indices_B = list(range(len(self)))
        random.shuffle(self.patch_indices_B)

    def __getitem__(self, index):
        """根据索引返回经过数据增强的 A 域和 B 域图像"""
        A_path = self.A_paths[0]
        B_path = self.B_paths[0]
        A_img = self.A_img
        B_img = self.B_img

        # 训练阶段，根据索引返回经过数据增强的 A 域和 B 域图像
        if self.opt.phase == "train":
            param = {'scale_factor': self.zoom_levels_A[index],
                     'patch_index': self.patch_indices_A[index],
                     'flip': random.random() > 0.5}

            transform_A = get_transform(self.opt, params=param, method=jt.transform.InterpolationMode.BILINEAR)
            A = transform_A(A_img)

            param = {'scale_factor': self.zoom_levels_B[index],
                     'patch_index': self.patch_indices_B[index],
                     'flip': random.random() > 0.5}
            transform_B = get_transform(self.opt, params=param, method=jt.transform.InterpolationMode.BILINEAR)
            B = transform_B(B_img)
        else:
            transform = get_transform(self.opt, method=jt.transform.InterpolationMode.BILINEAR)
            A = transform(A_img)
            B = transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """ 人为返回一个较大值（100000），模拟包含大量样本的数据集，实际是通过对单张图像的重复增强实现的"""
        return 10000

