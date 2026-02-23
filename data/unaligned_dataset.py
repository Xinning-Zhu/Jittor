import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import random
import util.util as util
from PIL import Image  

# 用于加载非对齐（unaligned/unpaired）数据集
class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        """构建图像路径,加载图像路径,记录数据集大小"""
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    
        self.A_size = len(self.A_paths)  
        self.B_size = len(self.B_paths)  

    def __getitem__(self, index):
        """根据索引返回域 A 和域 B 的图像数据及路径"""
        A_path = self.A_paths[index % self.A_size]  
        if self.opt.serial_batches:  
            index_B = index % self.B_size
        else:   
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB') 
        B_img = Image.open(B_path).convert('RGB')  

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform_func = get_transform(modified_opt)
        
        A = transform_func(A_img)
        B = transform_func(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """返回两个域中图像数量的最大值，确保遍历数据集时能覆盖所有样本"""
        return max(self.A_size, self.B_size)