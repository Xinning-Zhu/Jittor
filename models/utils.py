import jittor as jt
import os
import glob
import cv2
from PIL import Image
import numpy as np
from jittor import nn
import subprocess
import sys

# 添加RGB格式检查函数
def ensure_rgb_format(image_dir):
    """确保目录中的所有图像都是RGB格式"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(image_dir, filename)
            try:
                img = Image.open(filepath)
                if img.mode != 'RGB':
                    print(f"转换 {filename} 从 {img.mode} 到 RGB")
                    rgb_img = img.convert('RGB')
                    rgb_img.save(filepath)
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

# 使用PyTorch的torch_fidelity计算FID/KID
def calculate_fid_kid_with_torch(real_path, fake_path):
    """
    使用PyTorch的torch_fidelity计算FID和KID
    通过子进程调用Python解释器执行PyTorch代码
    """
    # 创建临时Python脚本
    torch_script = f"""
import torch
import torch_fidelity
import os

real_path = "{real_path}"
fake_path = "{fake_path}"

# 计算FID和KID
eval_args = {{
    'fid': True, 
    'kid': True, 
    'kid_subset_size': 50, 
    'kid_subsets': 10, 
    'verbose': False, 
    'cuda': True
}}

try:
    metrics = torch_fidelity.calculate_metrics(
        input1=real_path, 
        input2=fake_path, 
        **eval_args
    )
    fid = metrics['frechet_inception_distance']
    kid = metrics['kernel_inception_distance_mean'] * 100.0
    print(f"FID:{{fid}},KID:{{kid}}")
except Exception as e:
    print(f"ERROR:{{str(e)}}")
    print("FID:inf,KID:inf")
"""
    
    # 将脚本写入临时文件
    script_path = "/tmp/calculate_fid_kid.py"
    with open(script_path, 'w') as f:
        f.write(torch_script)
    
    try:
        # 执行PyTorch脚本
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=300)
        
        output = result.stdout.strip()
        
        if "ERROR" in output:
            print(f"PyTorch计算错误: {output}")
            return float('inf'), float('inf')
        
        # 解析结果
        if "FID:" in output and "KID:" in output:
            parts = output.split(',')
            fid = float(parts[0].split(':')[1])
            kid = float(parts[1].split(':')[1])
            return fid, kid
        else:
            print(f"无法解析结果: {output}")
            return float('inf'), float('inf')
            
    except subprocess.TimeoutExpired:
        print("FID/KID计算超时")
        return float('inf'), float('inf')
    except Exception as e:
        print(f"执行PyTorch脚本时出错: {e}")
        return float('inf'), float('inf')
    finally:
        # 清理临时文件
        if os.path.exists(script_path):
            os.remove(script_path)

# 评估流程封装
@jt.no_grad()
def eval_method(real_path, fake_path):
    print(f"真实图像路径: {real_path}")
    print(f"生成图像路径: {fake_path}")
    
    # 确保图像格式正确
    ensure_rgb_format(real_path)
    ensure_rgb_format(fake_path)
    
    eval_dict = {}
    try:
        # 计算FID和KID
        fid, kid = calculate_fid_kid_with_torch(real_path, fake_path)
        eval_dict['FID'] = fid
        eval_dict['KID'] = kid
        print(f"FID: {fid:.2f}, KID: {kid:.4f}")
    except Exception as e:
        print(f"计算FID/KID时出错: {e}")
        eval_dict['FID'] = float('inf')
        eval_dict['KID'] = float('inf')
    
    return eval_dict

# 图像保存
def save_image_numpy(image_numpy, image_path, aspect_ratio=1.0):
    """将numpy图像保存到磁盘，确保RGB格式"""
    # 确保是3通道
    if len(image_numpy.shape) == 2:  # 灰度图
        image_numpy = np.stack([image_numpy]*3, axis=-1)
    elif image_numpy.shape[2] == 1:  # 单通道
        image_numpy = np.repeat(image_numpy, 3, axis=2)
    elif image_numpy.shape[2] == 4:  # RGBA
        image_numpy = image_numpy[:, :, :3]  # 去掉alpha通道
    
    image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    
    image_pil.save(image_path)

# 跟踪训练 / 评估过程中的指标（如 FID、迭代次数）
class SimpleLogger:
    def __init__(self, path):
        self.path = path
    def log(self, iteration, max_iteration, metric_dict, verbose=False):
        message = '[%03d/%03d] ' % (iteration, max_iteration)
        for key in metric_dict:
            message += '\t %s:%.3f \t' % (key, metric_dict[key])
        if verbose:
            print(message)
        record = open(self.path, 'a')
        record.write('\n' + message + '\n')
        record.close()

    def log_message(self, message, verbose=True):
        record = open(self.path, 'a')
        record.write('\n' + message + '\n')
        if verbose:
            print(message)
        record.close()

# 处理模型输出的张量，转换为可保存的图像格式
def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, jt.Var):
            image_tensor = input_image
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    image_numpy = image_numpy.clip(0, 255)
    return image_numpy.astype(imtype)

# 批量处理测试集，生成假图像并保存到指定目录，再调用eval_method与真实图像对比，返回评估指标。
@jt.no_grad()
def eval_loader(model, test_loader_a, test_loader_b, output_directory, opt):
    fake_dir = os.path.join(output_directory, 'fake')
    if not os.path.exists(fake_dir):
        os.makedirs(fake_dir, exist_ok=True)
    
    if opt.direction == 'AtoB':
        test_loader = test_loader_a
        real_dir = os.path.join(opt.dataroot, 'testB')
        acc_loader = test_loader_b
    else:
        test_loader = test_loader_b
        real_dir = os.path.join(opt.dataroot, 'testA')
        acc_loader = test_loader_a

    for it, (data, acc_data) in enumerate(zip(test_loader, acc_loader)):
        fake = model.translate(data['A'])
        path_fake = os.path.join(fake_dir, os.path.basename(data['A_paths'][0]).replace('jpg', 'png'))
        im = tensor2im(fake)
        save_image_numpy(im, path_fake)
    
    eval_dict = eval_method(real_dir, fake_dir)
    return eval_dict

# 计算像素级误差指标：RMSE（均方根误差）和准确率（像素误差小于阈值的比例）
def eval_maps(real_path, fake_path, thr1=5, thr2=10, name=''):
    reals = glob.glob(real_path + '/*')
    fakes = glob.glob(fake_path + '/*')

    reals = sorted(reals)
    fakes = sorted(fakes)
    print(real_path, fake_path)

    num_imgs = len(reals)
    corr5_count = 0.0
    corr10_count = 0.0
    pix_count = 0.0
    RMSE = 0.0
    for i in range(num_imgs):

        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])

        real = cv2.resize(real, (256, 256), interpolation=cv2.INTER_LINEAR)
        fake = cv2.resize(fake, (256, 256), interpolation=cv2.INTER_LINEAR)

        real = real.ast(np.float32)
        fake = fake.astype(np.float32)
        diff = np.abs(real - fake)

        max_diff = np.max(diff, axis=2)

        corr5_count = corr5_count + np.sum(max_diff < thr1)
        corr10_count = corr10_count + np.sum(max_diff < thr2)
        pix_count = pix_count + 256**2

        diff = (diff**2) / (256**2)
        diff = np.sum(diff)
        rmse = np.sqrt(diff)
        RMSE = RMSE + rmse

    RMSE = RMSE / num_imgs
    acc5 = corr5_count / pix_count * 100.
    acc10 = corr10_count / pix_count * 100.
    eval_dict = {'%s/rmse' % (name): RMSE, '%s/acc@%d' % (name, thr1): acc5, '%s/acc@%d' % (name, thr2): acc10}
    return eval_dict

# 针对 Cityscapes 数据集的语义分割任务，评估生成的分割图与真实分割图的一致性
def eval_city2parsing(real_path, fake_path):
    labels = [{'name':'road', 'catId':0, 'color': (128, 64, 128)},
              {'name':'sidewalk', 'catId':1, 'color': (244, 35, 232)},
              {'name':'building', 'catId':2, 'color': (70, 70, 70)},
              {'name':'wall', 'catId':3, 'color': (102, 102, 156)},
              {'name':'fence', 'catId':4, 'color': (190, 153, 153)},
              {'name':'pole', 'catId':5, 'color': (153, 153, 153)},
              {'name':'traffic_light', 'catId':6, 'color': (250, 170, 30)},
              {'name':'traffic_sign', 'catId':7, 'color': (220, 220, 0)},
              {'name':'vegetation', 'catId':8, 'color': (107, 142, 35)},
              {'name':'terrain', 'catId':9, 'color': (152, 251, 152)},
              {'name':'sky', 'catId':10, 'color': (70, 130, 180)},
              {'name':'person', 'catId':11, 'color': (220, 20, 60)},
              {'name':'rider', 'catId':12, 'color': (255, 0, 0)},
              {'name':'car', 'catId':13, 'color': (0, 0, 142)},
              {'name':'truck', 'catId':14, 'color': (0, 0, 70)},
              {'name':'bus', 'catId':15, 'color': (0, 60, 100)},
              {'name':'train', 'catId':16, 'color': (0, 80, 100)},
              {'name':'motorcycle', 'catId':17, 'color': (0, 0, 230)},
              {'name':'bicycle', 'catId':18, 'color': (119, 11, 32)},
              {'name':'ignore', 'catId':19, 'color': (0, 0, 0)}]

    reals = glob.glob(real_path+'/*jpg')
    fakes = glob.glob(fake_path+'/*png')
    reals = sorted(reals)
    fakes = sorted(fakes)
    num_imgs = len(reals)

    CM = np.zeros((19,19), dtype=np.float32)
    for i in range(num_imgs):
        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])

        real = cv2.resize(real, (128, 128), interpolation=cv2.INTER_NEAREST)
        fake = cv2.resize(fake, (128, 128), interpolation=cv2.INTER_NEAREST)

        pred = fake
        label = real

        label_dis = np.zeros((20, 128, 128), dtype=np.float32)
        pred_dis = np.zeros((20, 128, 128), dtype=np.float32)

        for j in range(20):
            color = labels[j]['color']
            label_diff = np.abs(label - color)
            pred_diff = np.abs(pred - color)

            label_diff = np.sum(label_diff, axis=2)
            pred_diff = np.sum(pred_diff, axis=2)

            label_dis[j,:,:] = label_diff
            pred_dis[j,:,:] = pred_diff

        label_id = np.argmin(label_dis, axis=0)
        pred_id = np.argmin(pred_dis, axis=0)

        for j in range(19):
            coord = np.where(label_id == j)
            pred_j = pred_id[coord]
            for k in range(19):
                CM[j,k] = CM[j, k] + np.sum(pred_j == k)

    pix_acc = 0  # 整体像素准确率
    mean_acc = 0 # 各类别准确率的平均值
    mean_IoU = 0 # 各类别交并比（IoU）的平均值

    count = 0
    for i in range(19):
        count = count + CM[i, i]
    pix_acc = count / np.sum(CM)

    count = 0
    for i in range(19):
        temp = CM[i, :]
        count = count + CM[i,i]/(np.sum(temp) + 1e-6)
    mean_acc = count/19

    count = 0
    for i in range(19):
        temp_0 = CM[i, :]
        temp_1 = CM[:, i]
        count = count + CM[i, i]/(np.sum(temp_0) + np.sum(temp_1) - CM[i, i] + 1e-6)

    mean_IoU = count/19

    eval_dict = {'pix_acc':pix_acc, 'mean_acc':mean_acc, 'mean_IoU':mean_IoU}
    return eval_dict


if __name__ == '__main__':
    import sys
    out = eval_maps(sys.argv[1], sys.argv[2])
    print(out)


