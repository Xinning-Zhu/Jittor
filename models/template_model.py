"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import jittor as jt
from .base_model import BaseModel
from . import networks

# 实现了一个简单的端到端回归模型
class TemplateModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='aligned')  # 默认使用对齐数据集
        parser.add_argument('--lambda_regression', type=float, default=1.0, help='回归损失的权重')  

        return parser

    def __init__(self, opt):
        """初始化模型类

        参数:
            opt -- 训练/测试选项

        在此处可以完成以下工作:
        - 调用BaseModel的初始化函数
        - 定义损失函数、可视化图像、模型名称和优化器
        """
        BaseModel.__init__(self, opt)  
        self.loss_names = ['loss_G']
        self.visual_names = ['data_A', 'data_B', 'output']
        self.model_names = ['G']
        # 定义网络；可以使用opt.isTrain来指定训练和测试时的不同行为。
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=self.gpu_ids)
        if self.isTrain:  
            # 定义损失函数
            self.criterionLoss = jt.nn.L1Loss()
            # 定义和初始化优化器。可以为每个网络定义一个优化器。
            # 如果两个网络同时更新，可以使用itertools.chain将它们组合起来
            self.optimizer = jt.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer]


    def set_input(self, input):
        """从数据加载器中解包输入数据并执行必要的预处理步骤。

        参数:
            input: 一个字典，包含数据本身及其元数据信息。
        """
        AtoB = self.opt.direction == 'AtoB'  
        self.data_A = input['A' if AtoB else 'B']  
        self.data_B = input['B' if AtoB else 'A']  
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  

    def execute(self):
        """运行前向传播。将被<optimize_parameters>和<test>函数调用。"""
        self.output = self.netG(self.data_A)  # 给定输入data_A，生成输出图像

    def backward(self):
        """计算损失、梯度并更新网络权重；在每个训练迭代中调用"""
        # 给定输入和中间结果计算损失
        self.loss_G = self.criterionLoss(self.output, self.data_B) * self.opt.lambda_regression
        self.loss_G.backward()       # 计算网络G关于loss_G的梯度

    def optimize_parameters(self):
        """更新网络权重；将在每个训练迭代中调用。"""
        self.execute()               # 首先调用前向传播计算中间结果
        self.optimizer.zero_grad()   # 清除网络G现有的梯度
        self.backward()              # 计算网络G的梯度
        self.optimizer.step()        # 更新网络G的梯度
    