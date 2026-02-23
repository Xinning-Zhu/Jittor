"""
功能：
- 抽象基类 BaseDataset，统一数据集接口
- 实现 Jittor 版本的图像变换工具（替代 torchvision.transforms）
"""

import random
import numpy as np
from PIL import Image
import jittor as jt
from abc import ABC, abstractmethod

# 全局警告标记（图像尺寸需为4的倍数）
_print_size_warning_flag = False

class BaseDataset(jt.dataset.Dataset, ABC):
    """
    Jittor 数据集抽象基类
    子类需实现：
    - __len__: 返回数据集总样本数
    - __getitem__: 返回单样本数据（字典格式）
    - modify_commandline_options: （可选）添加数据集专属参数
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0  # 记录当前训练轮次

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        扩展命令行参数接口（如需添加数据集专属参数可在此实现）
        :param parser: Jittor 命令行参数解析器
        :param is_train: 是否为训练阶段
        :return: 修改后的解析器
        """
        return parser

    @abstractmethod
    def __len__(self):
        """返回数据集总样本数"""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """
        返回单样本数据
        :param index: 样本索引
        :return: dict, 包含数据及元信息（如 'image' 等键）
        """
        pass


def get_params(opt, size):
    """
    生成数据增强参数（裁剪位置、翻转标志）
    :param opt: 配置参数
    :param size: 图像原始尺寸 (w, h)
    :return: dict, 包含裁剪位置 (x, y) 和翻转标志 flip
    """
    w, h = size
    new_h, new_w = h, w

    # 预处理策略分支
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = int(opt.load_size * h / w)  

    # 随机裁剪位置
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))    
    flip = random.random() > 0.5  

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    """
    构建 Jittor 图像变换管道（替代 torchvision.transforms）
    :param opt: 配置参数
    :param params: 数据增强参数（裁剪位置、翻转等）
    :param grayscale: 是否转为灰度图
    :param method: 图像插值方法（默认双三次）
    :param convert: 是否转换为 Jittor 张量并归一化
    :return: 变换函数（接收 PIL Image，返回 Jittor 张量或 PIL Image） 
    """
    transform_funcs = []

    if grayscale:
        transform_funcs.append(lambda img: img.convert('L'))

    # 固定尺寸缩放
    if 'fixsize' in opt.preprocess:
        transform_funcs.append(lambda img: img.resize(params["size"], method))

    # resize 策略
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        if "gta2cityscapes" in opt.dataroot:
            osize[0] = opt.load_size // 2
        transform_funcs.append(lambda img: img.resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_funcs.append(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method))
    elif 'scale_shortside' in opt.preprocess:
        transform_funcs.append(lambda img: __scale_shortside(img, opt.load_size, opt.crop_size, method))

    # 随机缩放
    if 'zoom' in opt.preprocess:
        if params is None:
            transform_funcs.append(lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method))
        else:
            transform_funcs.append(
                lambda img: __random_zoom(img, opt.load_size, opt.crop_size, method, factor=params["scale_factor"])
            )

    # 裁剪
    if 'crop' in opt.preprocess:
        if params is None or 'crop_pos' not in params:
            def random_crop(img):
                ow, oh = img.size
                x = random.randint(0, max(0, ow - opt.crop_size))
                y = random.randint(0, max(0, oh - opt.crop_size))
                return img.crop((x, y, x + opt.crop_size, y + opt.crop_size))
            transform_funcs.append(random_crop)
        else:
            transform_funcs.append(lambda img: __crop(img, params['crop_pos'], opt.crop_size))

    # 补丁裁剪（按网格索引）
    if 'patch' in opt.preprocess:
        transform_funcs.append(lambda img: __patch(img, params['patch_index'], opt.crop_size))

    # 边缘裁剪
    if 'trim' in opt.preprocess:
        transform_funcs.append(lambda img: __trim(img, opt.crop_size))

    transform_funcs.append(lambda img: __make_power_2(img, base=4, method=method))

    # 水平翻转
    if not opt.no_flip:
        if params is None or 'flip' not in params:
            transform_funcs.append(lambda img: __flip(img, random.random() > 0.5))
        elif 'flip' in params:
            transform_funcs.append(lambda img: __flip(img, params['flip']))

    # 转换为 Jittor 张量并归一化到[-1, 1]范围
    if convert:
        def to_jittor_tensor(img):
            """将 PIL Image 转换为 Jittor 张量并归一化"""
            if grayscale:
                # 灰度图: [H, W] -> [1, H, W]
                tensor = jt.array(np.array(img)).unsqueeze(0).float32()
            else:
                # 彩色图: [H, W, C] -> [C, H, W]
                tensor = jt.array(np.array(img)).permute(2, 0, 1).float32()
            # 归一化到 [-1, 1]
            return (tensor / 127.5) - 1.0

        transform_funcs.append(to_jittor_tensor)

    # 组合变换函数
    def compose(img):
        for func in transform_funcs:
            img = func(img)
        return img

    return compose


# ------------------------- 以下为基础变换函数 ------------------------- #
def __make_power_2(img, base, method=Image.BICUBIC):
    """调整图像尺寸为 base 的倍数（确保能被 4 整除等场景）"""
    global _print_size_warning_flag
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)

    if not _print_size_warning_flag and (ow != w or oh != h):
        print(f"The image size needs to be a multiple of {base}. The original size ({ow}, {oh}) has been adjusted to ({w}, {h}).")
        _print_size_warning_flag = True

    return img.resize((w, h), method) if (w, h) != (ow, oh) else img


def __random_zoom(img, target_width, crop_width, method=Image.BICUBIC, factor=None):
    """随机缩放图像"""
    if factor is None:
        # 随机缩放因子 [0.8, 1.0]
        zoom_level = np.random.uniform(0.8, 1.0, size=2)
    else:
        zoom_level = factor

    iw, ih = img.size
    # 计算缩放后的尺寸（确保不小于裁剪尺寸）
    zoomw = max(crop_width, int(iw * zoom_level[0]))
    zoomh = max(crop_width, int(ih * zoom_level[1]))

    return img.resize((zoomw, zoomh), method)


def __scale_shortside(img, target_width, crop_width, method=Image.BICUBIC):
    """缩放到短边为 target_width（保持宽高比）"""
    ow, oh = img.size
    shortside = min(ow, oh)
    if shortside >= target_width:
        return img
    # 计算缩放比例
    scale = target_width / shortside
    return img.resize((int(round(ow * scale)), int(round(oh * scale))), method)


def __trim(img, trim_width):
    """随机裁剪边缘（得到 trim_width 尺寸的图像）"""
    ow, oh = img.size
    # 计算裁剪区域
    x_start = random.randint(0, max(0, ow - trim_width)) if ow > trim_width else 0
    y_start = random.randint(0, max(0, oh - trim_width)) if oh > trim_width else 0
    x_end = x_start + trim_width
    y_end = y_start + trim_width

    return img.crop((x_start, y_start, x_end, y_end))


def __scale_width(img, target_width, crop_width, method=Image.BICUBIC):
    """缩放到固定宽度（保持宽高比，确保高度不小于裁剪尺寸）"""
    ow, oh = img.size
    if ow == target_width and oh >= crop_width:
        return img
    # 计算目标高度
    h = int(max(target_width * oh / ow, crop_width))
    return img.resize((target_width, h), method)


def __crop(img, pos, size):
    """固定位置裁剪"""
    ow, oh = img.size
    x1, y1 = pos
    tw, th = size
    if ow > tw or oh > th:
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __patch(img, index, size):
    """按网格索引裁剪补丁（用于多尺度训练等场景）"""
    ow, oh = img.size
    nw, nh = ow // size, oh // size  
    roomx = ow - nw * size  
    roomy = oh - nh * size  

    # 随机偏移
    startx = random.randint(0, roomx) if roomx > 0 else 0
    starty = random.randint(0, roomy) if roomy > 0 else 0

    # 计算网格坐标
    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size

    return img.crop((gridx, gridy, gridx + size, gridy + size))


def __flip(img, flip):
    """水平翻转"""
    return img.transpose(Image.FLIP_LEFT_RIGHT) if flip else img

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True