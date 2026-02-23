from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image

class SingleDataset(BaseDataset):
    """从指定目录加载单一域的图像数据（而非成对的图像），并对其进行预处理，以便在生成模型的测试阶段使用。
       该类可单独加载某一域的图像，用于生成另一域的转换结果。
    """

    def __init__(self, opt):
        """继承并调用父类BaseDataset的初始化方法，获取配置参数opt"""
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.dataroot, opt.max_dataset_size))
        # 根据配置确定输入图像的通道数
        input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.transform = get_transform(opt, grayscale=(input_nc == 1))

    def __getitem__(self, index):
        """根据索引返回单张图像的数据及路径"""
        A_path = self.A_paths[index]
        A_img = Image.open(A_path).convert('RGB')
        A = self.transform(A_img)
        return {'A': A, 'A_paths': A_path}

    def __len__(self):
        """直接返回数据集中图像的总数量"""
        return len(self.A_paths)
