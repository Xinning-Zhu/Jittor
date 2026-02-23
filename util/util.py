"""This module contains simple helper functions """
from __future__ import print_function
import jittor as jt
import jittor.transform as jt_transform
import jittor.misc as vutils
import numpy as np
from PIL import Image
import os
import importlib
import argparse
from argparse import Namespace
from fnmatch import fnmatch
import shutil
import hashlib


#参数类型转换与配置处理
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def copyconf(default_opt, **kwargs):
    conf = Namespace(** vars(default_opt))
    for key in kwargs:
        setattr(conf, key, kwargs[key])
    return conf

# 类与模块查找
def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    assert cls is not None, "In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name)

    return cls

# 张量与图像转换
def tensor2im(input_image, imtype=np.uint8):
    """将 Jittor 张量转换为 NumPy 图像数组，
        支持灰度图转 RGB，并将取值范围从[-1, 1]映射到[0, 255]，
        用于将网络输出的张量可视化为图像。
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, jt.Var):  
            image_tensor = input_image  
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()
        if image_numpy.shape[0] == 1:  
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:  
        image_numpy = input_image
    return image_numpy.astype(imtype)

# 网络诊断与信息打印
def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (Jittor network) -- Jittor network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += jt.mean(jt.abs(param.grad))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

# 将 NumPy 图像数组保存为图像文件，支持按比例调整尺寸，用于保存生成的图像结果。
def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

# 信息打印
def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

# 创建目录
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

# 调整标签张量的大小（使用最近邻插值），适用于语义分割等任务中标签与图像尺寸匹配。
def correct_resize_label(t, size):
    t = t.detach() 
    resized = []
    for i in range(t.shape[0]): 
        one_t = t[i, :1] 
        one_np = np.transpose(one_t.numpy().astype(np.uint8), (1, 2, 0))
        one_np = one_np[:, :, 0]  
        one_image = Image.fromarray(one_np).resize(size, Image.NEAREST)
        resized_t = jt.array(np.array(one_image)).long()
        resized.append(resized_t)
    return jt.stack(resized, dim=0)

# 调整图像张量的大小（默认双三次插值），并转换为 Jittor 张量格式，用于数据预处理或中间结果的尺寸统一。
def correct_resize(t, size, mode=Image.BICUBIC):
    t = t.detach()
    resized = []
    for i in range(t.size(0)):
        one_t = t[i:i + 1]
        one_image = Image.fromarray(tensor2im(one_t)).resize(size, Image.BICUBIC)
        resized_t = jt_transform.to_tensor(one_image) * 2 - 1.0
        resized.append(resized_t)
    return jt.stack(resized, dim=0)


watched_rules = ['*.py', '*.sh', '*.yaml', '*.yml']
exclude_rules = ['results', 'datasets', 'checkpoints', 'samples', 'outputs']
def calculate_checksum(filenames):
    hash = hashlib.md5()
    for fn in filenames:
        if os.path.isfile(fn):
            hash.update(open(fn, "rb").read())
    return hash.hexdigest()


def copy_files_and_create_dirs(files, target_dir):
    """Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories."""
    if len(files) >= 500:
        print('Warning! there are %d files to be copied!' %(len(files)))
    for file in files:
        target_name = os.path.join(target_dir, file)
        dir_name = os.path.dirname(target_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # will create all intermediate-level directories
        shutil.copyfile(file, target_name)


def _get_watched_files(work_dir):
    rules = watched_rules
    watched_files = []
    to_match = []
    for rule in rules:
        t = rule.count('*')
        if t == 0:
            watched_files.append(rule)
        elif t == 1:
            to_match.append(rule)

    for parent, dirs, file_names in os.walk(work_dir):
        for ignore_ in exclude_rules:
            dirs_to_remove = [d for d in dirs if fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            file_names = [f for f in file_names if not fnmatch(f, ignore_)]

        for file_name in file_names:
            for each in to_match:
                if fnmatch(file_name, each):
                    watched_files.append(os.path.join(parent, file_name))
                    break
    return watched_files


def prepare_sub_directories(run_dir, opt):

    src_dir = os.path.join(run_dir, 'src')
    files = _get_watched_files('.')
    copy_files_and_create_dirs(files, src_dir)
    opt.src_dir = src_dir

    img_dir = os.path.join(run_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    opt.img_dir = img_dir

# 将多个图像张量拼接成网格并保存为单张图像，支持扩展灰度图到 3 通道，用于批量可视化网络输出（如生成器的多个结果对比）。
def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]
    image_tensor = jt.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor, nrow=display_image_num, padding=0, normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_images(image_outputs, display_image_num, image_directory, postfix):
    __write_images(image_outputs, display_image_num, '%s/gen_%s.jpg' % (image_directory, postfix))