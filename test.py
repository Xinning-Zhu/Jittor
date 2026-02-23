"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model  
from util.visualizer import save_images
from util import html
import util.util as util
import jittor as jt  
import time


jt.flags.use_cuda = 1 if jt.has_cuda else 0
if __name__ == '__main__':
    opt = TestOptions().parse() 
    opt.num_threads = 0        # 测试代码仅支持单线程
    opt.batch_size = 1         # 测试代码仅支持batch_size=1
    opt.serial_batches = True  # 禁用数据打乱
    opt.no_flip = True         # 禁用图像翻转
    opt.display_id = -1        # 不使用visdom显示，结果保存到HTML
    
    dataset = create_dataset(opt)                                      # 创建测试数据集
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))  # 创建训练数据集
    model = create_model(opt)                                          # 根据参数创建模型
    
    # 创建结果网页目录
    web_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}')
    print('创建网页目录:', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # 确定最大测试样本数
    if opt.direction == 'AtoB':
        max_num_test = len(os.listdir(os.path.join(opt.dataroot, 'testA')))
    else:
        max_num_test = len(os.listdir(os.path.join(opt.dataroot, 'testB')))
    
    loss_pathes = []  
    for i, data in enumerate(dataset):
        if i == 0:
            # 模型初始化
            model.data_dependent_initialize(data)
            model.setup(opt)  
            if opt.eval:
                model.eval()  
        
        if i >= opt.num_test:  
            break
        
        model.set_input(data) 
        st = time.time()
        model.forward()  # 执行推理
        et = time.time()
        print(f"第{i}张图像推理耗时: {et - st:.4f}秒")  # 打印单张图像推理时间
        
        visuals = model.get_current_visuals()  # 获取可视化结果
        img_path = model.get_image_paths()     # 获取图像路径
        
        if i % 5 == 0:  # 每5张图像保存一次到HTML
            print('processing (%04d)-th image... %s' % (i, img_path))
        
        save_images(webpage, visuals, img_path, width=opt.display_winsize)
    
    webpage.save()  # 保存HTML文件
    print(f"测试结果已保存至: {web_dir}")