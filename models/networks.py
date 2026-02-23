import jittor as jt
import jittor.nn as nn
import numpy as np
import functools
from jittor import lr_scheduler
from .stylegan_networks import StyleGAN2Discriminator, StyleGAN2Generator, TileStyleGAN2Discriminator
from . import networks 
# 生成高斯滤波核，用于在采样时平滑图像，减少锯齿效应
def get_filter(filt_size=3):
    # 根据滤波核大小生成二项式系数（近似高斯分布）
    if filt_size == 1:
        a = np.array([1., ])
    elif filt_size == 2:
        a = np.array([1., 1.])
    elif filt_size == 3:
        a = np.array([1., 2., 1.])
    elif filt_size == 4:
        a = np.array([1., 3., 3., 1.])
    elif filt_size == 5:
        a = np.array([1., 4., 6., 4., 1.])
    elif filt_size == 6:
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif filt_size == 7:
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    # 生成二维滤波核（外积）并归一化
    filt = jt.array(a[:, None] * a[None, :])
    filt = filt / jt.sum(filt)
    return filt

# 生成填充层，用于在采样前扩展图像边缘，避免边缘信息丢失
def get_pad_layer(pad_type):
    # 根据填充类型返回对应的填充层生成函数
    if pad_type in ['refl', 'reflect']:
        def pad_layer(padding):
            if isinstance(padding, (list, tuple)):
                if all(p == padding[0] for p in padding):
                    return nn.ReflectionPad2d(padding[0])
                else:
                    return nn.ZeroPad2d(padding)
            else:
                return nn.ReflectionPad2d(padding)
        return pad_layer
    elif pad_type in ['repl', 'replicate']:
        def pad_layer(padding):
            if isinstance(padding, (list, tuple)):
                if all(p == padding[0] for p in padding):
                    return nn.ReplicationPad2d(padding[0])
                else:
                    return nn.ZeroPad2d(padding)
            else:
                return nn.ReplicationPad2d(padding)
        return pad_layer
    elif pad_type == 'zero':
        def pad_layer(padding):
            return nn.ZeroPad2d(padding)
        return pad_layer
    else:
        raise ValueError(f'不支持的填充类型: {pad_type}')

# 下采样模块
class Downsample(nn.Module):
    """对输入图像进行填充，避免边缘像素在滤波时被过度裁剪。
       使用预生成的滤波核进行分组卷积，步长为stride（通常为 2），实现下采样。
       相比简单的nn.AvgPool2d，该方法通过平滑滤波减少下采样带来的锯齿和混叠，保留更多细节。
    """
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            max(0, int(1. * (filt_size - 1) / 2)),
            max(0, int(np.ceil(1. * (filt_size - 1) / 2))),
            max(0, int(1. * (filt_size - 1) / 2)),
            max(0, int(np.ceil(1. * (filt_size - 1) / 2)))
        ]
        self.pad_sizes = tuple([max(0, pad_size + pad_off) for pad_size in self.pad_sizes])
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))  

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def execute(self, inp):  
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            padded_inp = self.pad(inp)
            return nn.conv2d(padded_inp, self.filt, stride=self.stride, groups=inp.shape[1])

# 轻量级上采样，基于插值实现
class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode if mode in ['nearest', 'bilinear'] else 'nearest'

    def execute(self, x):  
        return nn.interpolate(x, scale_factor=self.factor, mode=self.mode)

# 高质量上采样，基于转置卷积和滤波核实现
class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.filt = filt[None, None, :, :].repeat((self.channels, 1, 1, 1))  
        self.pad = get_pad_layer(pad_type)(1) 

    def execute(self, inp):  
        padded_inp = self.pad(inp)
        ret_val = nn.conv_transpose2d(
            padded_inp, 
            self.filt, 
            stride=self.stride, 
            padding=1 + self.pad_size, 
            groups=inp.shape[1]
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]
        

class Identity(nn.Module):
    def execute(self, x):  
        return x

# 归一化层配置
def get_norm_layer(norm_type='instance'):
    # 批归一化，适用于较大批量的训练，能加速收敛并提高稳定
    if norm_type == 'batch':
        norm_layer = functools.partial(jt.nn.BatchNorm2d, affine=True, track_running_stats=True)
    # 实例归一化
    elif norm_type == 'instance':
        norm_layer = functools.partial(jt.nn.InstanceNorm2d, affine=False)
    # 返回恒等映射层，不需要归一化操作
    elif norm_type == 'none':
        def norm_layer(x): 
            return Identity() 
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# 学习率调度器
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use custom implementations for Jittor.
    """
    # 线性衰减 ，前面保持初始学习率，后面再进行衰减到0
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        
        class LinearLR:
            def __init__(self, optimizer, lambda_rule):
                self.optimizer = optimizer
                self.lambda_rule = lambda_rule
                self.epoch = 0
                
            def step(self):
                self.epoch += 1
                lr = self.lambda_rule(self.epoch)
                for param_group in self.optimizer.param_groups:
                    initial_lr = param_group.get('initial_lr', opt.lr)
                    param_group['lr'] = initial_lr * lr
        
        scheduler = LinearLR(optimizer, lambda_rule)# 创建调度器实例，后续在训练循环中通过调用scheduler.step()来更新学习率
    
    # 阶梯衰减，每经过opt.lr_decay_iters个epoch，学习率乘以 0.1
    elif opt.lr_policy == 'step':
        scheduler = jt.optim.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    # jt.optim.StepLR：这是 Jittor 框架（一个深度学习框架）提供的 StepLR 调度器类，功能是按照固定步数对学习率进行衰减
    # optimizer：需要调整学习率的优化器（如 SGD、Adam 等）

    # 当损失等不再改善时衰减学习率
    elif opt.lr_policy == 'plateau':
        """持续监控某个性能指标（如损失值或准确率）
        当指标在一定轮次内没有明显改善时，自动降低学习率
        这种策略适用于无法预先确定衰减时机的场景，更具自适应能力
        """
        class ReduceLROnPlateau:
            # mode：指标优化方向（'min'表示希望指标越小越好，如损失值；'max'表示希望指标越大越好，如准确率）
            # threshold：判断指标是否改善的阈值（只有当指标变化超过这个阈值时，才认为有改善）
            # patience："容忍" 的轮次（连续多少轮指标无改善后，才触发学习率衰减）
            def __init__(self, optimizer, mode='min', factor=0.2, threshold=0.01, patience=5):
                self.optimizer = optimizer
                self.mode = mode
                self.factor = factor
                self.threshold = threshold
                self.patience = patience
                self.best = None
                self.num_bad_epochs = 0
                
            def step(self, metrics):
                if self.best is None:
                    self.best = metrics
                    return
                
                if self.mode == 'min':
                    is_better = metrics < self.best - self.threshold
                else:  
                    is_better = metrics > self.best + self.threshold
                
                if is_better:
                    self.best = metrics
                    self.num_bad_epochs = 0
                else:
                    self.num_bad_epochs += 1
                
                if self.num_bad_epochs >= self.patience:
                    self.num_bad_epochs = 0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.factor
        
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    
    elif opt.lr_policy == 'cosine':
        # 余弦退火,T_max：余弦退火的周期（单位：epoch），表示学习率从初始值衰减到最小值所需的总轮次,eta_min：学习率的最小值（默认为 0）

        class CosineAnnealingLR:
            def __init__(self, optimizer, T_max, eta_min=0):
                self.optimizer = optimizer
                self.T_max = T_max
                self.eta_min = eta_min
                self.epoch = 0
                
            def step(self):
                self.epoch += 1
                # 计算余弦衰减的学习率
                lr = self.eta_min + 0.5 * (self.optimizer.param_groups[0].get('initial_lr', opt.lr) - self.eta_min) * \
                     (1 + math.cos(math.pi * self.epoch / self.T_max))
                # 更新所有参数组的学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
        
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    
    return scheduler



# 初始化神经网络权重的函数 init_weights
# init_gain：初始化时的增益参数
def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights"""
    def init_func(m): 
        classname = m.__class__.__name__
        # 筛选出包含权重参数（weight）的层：卷积层（Conv）、全连接层（Linear）、嵌入层（Embedding）
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1
                                     or classname.find('Embedding') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                # 正态分布初始化
                nn.init.gauss_(m.weight, 0.0, init_gain) # 均值0，标准差为init_gain的高斯分布
            elif init_type == 'xavier': # 适用于激活函数为tanh/sigmoid
                # 尝试Xavier正态分布，若不支持则 fallback 到 Kaiming初始化
                try:
                    # 先尝试使用 xavier_normal_（如果某些版本有）
                    nn.init.xavier_normal_(m.weight, gain=init_gain)
                except AttributeError:
                    # # 适用于ReLU等激活函数，解决梯度消失问题
                    nn.init.kaiming_normal_(m.weight)
            elif init_type == 'kaiming':
                # 适用于ReLU等激活函数，解决梯度消失问题
                # Jittor 的 kaiming_normal_ 参数略有不同
                nn.init.kaiming_normal_(m.weight)
            elif init_type == 'orthogonal':
                # Jittor 支持 orthogonal_
                # 使权重矩阵列向量正交，减轻梯度消失
                nn.init.orthogonal_(m.weight, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            # 偏置项（bias）初始化
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0) # 统一初始化为0
        elif classname.find('BatchNorm') != -1:  # 修改为更通用的 BatchNorm 检测
            # Jittor 使用 gauss_ 而不是 normal_
            nn.init.gauss_(m.weight, 1.0, init_gain) # 缩放参数初始化为均值1的高斯分布
            nn.init.constant_(m.bias, 0.0) # 偏移参数初始化为0

    # Jittor 使用 apply 方法，不需要手动遍历 modules
    net.apply(init_func)

# 初始化神经网络，可以让神经网络初始化过程更加简洁，只要调用一个函数即可完成模型训练前的准备工作，提高代码的复用性和易用性
def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
# 神经网络初始化过程的上层封装，整合了权重初始化和设备配置（GPU/CPU）的功能
# debug：是否打印调试信息（传递给init_weights函数）

    if len(gpu_ids) > 0:
        jt.flags.use_cuda = True  # Jittor自动管理GPU，通过flags设置
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


#用于创建生成器网络的函数，根据传入的参数选择不同类型的生成器架构，并完成初始化配置。这在生成对抗网络（GAN）等场景中非常常见
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    # ngf：生成器第一个卷积层的滤波器数量（控制网络宽度）
    # netG：生成器的架构类型（如'resnet_9blocks'、'unet_256'等）
    # norm：归一化层类型（如'batch'表示批归一化）
    # use_dropout：是否使用 dropout 层（防止过拟合）
    # init_type/init_gain：权重初始化方法和增益参数
    # no_antialias/no_antialias_up：是否禁用抗锯齿处理（影响下采样 / 上采样质量）

    net = None
    # 根据norm参数获取对应的归一化层，归一化层是稳定网络训练的重要组件
    norm_layer = get_norm_layer(norm_type=norm)

    # 'resnet_9blocks'/'resnet_6blocks'/'resnet_4blocks'：分别使用 9/6/4 个残差块的 ResNet 架构，残差块能有效缓解深层网络的梯度消失问题
    # 'adain_resnet_9blocks'等：结合了 AdaIN（自适应实例归一化）的 ResNet，常用于风格迁移任务
    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt)
    elif netG == 'adain_resnet_9blocks':
        net = AdaINResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'multiadain_resnet_9blocks':
        net = MultiAdaINResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                   no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9, opt=opt)
    elif netG == 'adain_resnet_6blocks':
        net = AdaINResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                  no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt)
    elif netG == 'resnet_4blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4, opt=opt)
    
    #'unet_128'/'unet_256'：输入图像尺寸分别为 128×128 和 256×256 的 U-Net 架构，通过编码器 - 解码器结构实现端到端生成
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    
    # 'stylegan2'/'smallstylegan2'：基于 StyleGAN2 的生成器，擅长生成高质量图像，支持风格控制
    # 'smallstylegan2'是简化版，使用更少的残差块
    elif netG == 'stylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, opt=opt)
    elif netG == 'smallstylegan2':
        net = StyleGAN2Generator(input_nc, output_nc, ngf, use_dropout=use_dropout, n_blocks=2, opt=opt)
    
    # 'resnet_cat'：一种拼接（concatenate）特征的 ResNet 变体
    elif netG == 'resnet_cat':
        n_blocks = 8
        net = G_Resnet(input_nc, output_nc, opt.nz, num_downs=2, n_res=n_blocks - 4, ngf=ngf, norm='inst', nl_layer='relu')
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))


# 用于创建特征提取 / 投影网络，它根据参数选择不同类型的特征处理网络，并并完成初始化
def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
             no_antialias=False, gpu_ids=[], opt=None, num_params=None):
    # 全局池化网络，用于将空间特征压缩为全局特征向量
    if netF == 'global_pool':
        net = PoolingF()
    # 重塑网络，可能用于调整特征的维度或形状（如将高维特征重塑为特定格式）
    elif netF == 'reshape':
        net = ReshapeF()
    # 多层感知机，用于将特征映射到另一个空间（如风格向量映射），dim指定特征维度，num_params控制参数规模。
    elif netF == 'mlp':
        net = MappingNetwork(dim=opt.netF_nc, num_params=num_params, opt=opt)
    #'sample'/'mlp_sample'（特征采样网络），用于从特征图中采样局部 patch 特征，use_mlp=False表示直接采样，use_mlp=True表示采样后通过 MLP 进一步处理，常用于对比学习或特征匹配。
    elif netF == 'sample':
        net = PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    # 用于判断特征来自哪个领域（如领域适应任务），通过局部 patch 特征进行领域分类
    elif netF == 'domain_classifier':
        net = PatchDomainClassifier(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    # 使用带步长的卷积层实现特征下采样和提取，通过卷积操作压缩空间维度并保留关键特征
    elif netF == 'strided_conv':
        net = StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return init_net(net, init_type, init_gain, gpu_ids)


# 用于创建判别器（Discriminator）网络的函数，主要用于生成对抗网络（GAN）中区分真实数据与生成数据
def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    # n_layers_D：判别器的卷积层数（用于灵活调整网络深度）
    # ndf：判别器第一个卷积层的滤波器数量（控制网络宽度）
    # netD：判别器的架构类型（决定网络结构和功能）

    
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    # 经典的 PatchGAN 结构，输出一个特征图而非单个值，通过判断图像局部区域（patch）的真实性来提升生成细节质量，默认使用 3 层卷积。
    if netD == 'basic':  
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, no_antialias=no_antialias,)
    # 带实例归一化的全局判别器，使用6层卷积，对整个图像进行全局判断，适合需要全局一致性的任务，使用更多卷积层提升特征提取能力
    elif netD == 'ins_global':  
        net = GlobalDiscriminator(input_nc, ndf, n_layers=6, norm_layer=norm_layer, no_antialias=no_antialias, )
    # 全局图像判别器，专注于判断整个图像的真实性，输出单个概率值（真实 / 生成），适用于需要全局评估的场景
    elif netD == 'global': 
        net = GlobalImageDis(input_nc)
    # 自定义层数的判别器，灵活指定卷积层数，便于调整网络深度进行实验
    elif netD == 'n_layers':  
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, no_antialias=no_antialias,)
    # 像素级判别器，对图像的每个像素进行真实性判断，输出与输入尺寸相同的概率图，适合需要精细像素级控制的任务
    elif netD == 'pixel':    
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    # 基于 StyleGAN2 的判别器结构，通常配合 StyleGAN2 生成器使用，擅长处理高分辨率图像
    elif 'stylegan2' in netD:
        net = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias=no_antialias, opt=opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids,
                    initialize_weights=('stylegan2' not in netD))

# 计算生成对抗网络（GAN）中不同类型的损失函数，是 GAN 训练的核心组件之一
class GANLoss(nn.Module):
    # 配置损失模式与基础参数，target_real_label：真实样本的标签值（默认 1.0）
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        # 替换register_buffer为直接定义属性（Jittor自动管理非参数张量）
        # 用 Jittor 的jt.array创建张量（替代 PyTorch 的torch.tensor），Jittor 会自动管理这些非参数张量的设备（CPU/GPU）。
        self.real_label = jt.array(target_real_label)
        self.fake_label = jt.array(target_fake_label)
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    # 生成匹配尺寸的标签张量,根据判别器的预测结果（prediction）尺寸，生成对应大小的标签张量（真实 / 生成标签），确保标签与预测结果维度匹配
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.broadcast(prediction.shape)  # 替换expand_as为broadcast
        #用 Jittor 的broadcast方法（替代 PyTorch 的expand_as）将标量标签张量扩展为与prediction相同的形状（如prediction是[batch_size, 1, 32, 32]，标签也会扩展为该尺寸）。

    # 核心计算方法（execute）：计算最终 GAN 损失
    # Jittor 中用execute方法替代 PyTorch 的forward，作为模块的前向传播入口，接收判别器的预 替换forward为execute测结果和标签类型，返回最终损失值。
    def execute(self, prediction, target_is_real):  
        bs = prediction.shape[0] 
        # 'lsgan'或'vanilla'模式：基于预绑定损失函数
        if self.gan_mode in ['lsgan', 'vanilla']:
            # 先调用get_target_tensor生成匹配标签，再用预绑定的self.loss（MSE 或 BCEWithLogits）计算预测值与标签的损失。
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        # WGAN-GP 损失（无标签，直接基于预测均值），WGAN-GP 通过限制判别器为 1-Lipschitz 函数，损失无需标签，直接基于预测值的均值调整，避免传统 GAN 的梯度消失问题。
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        # 'nonsaturating'模式：非饱和 GAN 损失（基于 softplus 的自定义计算）
        # 核心目的：通过非饱和损失设计，让生成器在训练后期仍能获得有效梯度，避免梯度消失。
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                # nn.softplus：替代sigmoid的平滑激活函数，计算softplus(x) = ln(1 + e^x)，避免梯度饱和；
                # reshape(bs, -1)：将预测结果展平为[batch_size, 特征总数]（替代 PyTorch 的view）；
                # mean(dim=1)：对每个样本的所有特征求均值，得到每个样本的损失值（便于后续按批次处理）。
                loss = nn.softplus(-prediction).reshape(bs, -1).mean(dim=1)  # 替换F.softplus为nn.softplus，view为reshape
            else:
                loss = nn.softplus(prediction).reshape(bs, -1).mean(dim=1)  # 替换F.softplus为nn.softplus，view为reshape
        return loss

# 梯度惩罚损失，WGAN-GP 改用 “梯度惩罚” 替代权重裁剪：通过对 “真实样本与生成样本之间的插值样本” 计算判别器的梯度，强制梯度的 L2 范数接近 1，既满足约束又避免权重扭曲，让 GAN 训练更稳定。
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    # type	插值样本类型（'real'/'fake'/'mixed'，默认'mixed'为真实与生成的插值）
    # constant	梯度 L2 范数的目标值（1-Lipschitz 约束要求梯度范数≤1，故默认1.0）
    # lambda_gp	梯度惩罚的权重系数（控制梯度惩罚在总损失中的占比，默认10.0为经典值）
    # 函数按 “生成插值样本 → 计算判别器对插值的梯度 → 计算梯度惩罚损失” 三步执行
    if lambda_gp > 0.0:
        # 根据type参数选择插值样本的来源，核心是构建 “真实样本与生成样本之间的连续过渡样本”，这是梯度惩罚的计算基础
        if type == 'real':
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            # 核心：生成真实与生成样本的随机插值（最常用，满足1-Lipschitz约束的关键）
            alpha = jt.rand(real_data.shape[0], 1, device=device)  # 生成[0,1)的随机系数
            alpha = alpha.broadcast(real_data.shape)  # 替换expand为broadcast,扩展到与样本相同形状（如[batch,3,256,256]）
            # 插值公式：interpolate = α*real + (1-α)*fake（α为每个样本的独立随机系数）
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        # 第二步：计算判别器对插值样本的梯度
        interpolatesv.requires_grad = True # 1. 标记插值样本需要计算梯度（关键：Jittor默认不追踪梯度）
        disc_interpolates = netD(interpolatesv) # 2. 判别器对插值样本的预测输出
        # 替换torch.autograd.grad为jt.grad，计算梯度
        gradients = jt.grad(
            outputs=disc_interpolates, # 目标输出（判别器预测）
            inputs=interpolatesv,# 待求梯度的输入（插值样本）
            grad_outputs=jt.ones(disc_interpolates.shape).to(device),  # 梯度初始值（全1，等价于对输出求导）# 替换torch.ones为jt.ones
            retain_graph=True, # 保留计算图（后续可能还有其他梯度计算，避免图被销毁）
            create_graph=True  # 创建二阶导数图（梯度惩罚损失需对梯度本身求导，故需二阶图）
        )[0] # jt.grad返回梯度列表，取第一个元素（对应inputs=interpolatesv的梯度）
        # 第三步：计算梯度惩罚损失
        # 1. 将梯度展平为[batch_size, 总特征数]（如[32, 3*256*256]），便于按样本计算L2范数
        gradients = gradients.reshape(real_data.shape[0], -1)  # 替换view为reshape
        # 2. 梯度惩罚公式：(||∇_x D(x)||₂ - 1)² 的均值 × 权重lambda_gp
        # +1e-16是为了避免梯度范数为0时开方出错（数值稳定性）
        # (gradients + 1e-16).norm(2, dim=1)：计算每个样本梯度的 L2 范数（按 dim=1 对展平后的特征求范数）；
        #(范数 - 1)²：衡量梯度范数与 1-Lipschitz 约束目标（1.0）的偏差，偏差越大惩罚越重；
        #.mean()：对批次内所有样本的偏差取均值；
        #× lambda_gp：乘以权重系数，控制梯度惩罚在判别器总损失中的贡献。
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp
        return gradient_penalty, gradients
    else:
        return 0.0, None

# 用于对输入张量进行L-p 范数归一化，即欧几里得范数归一化
# 其核心作用是将张量的特征沿指定维度缩放，使其范数（“长度”）满足特定条件，常用于特征标准化、相似度计算等场景
class Normalize(nn.Module):
    # 本质是一个 “特征归一化层”，可以像卷积层、激活层一样嵌入到神经网络中，对经过该层的张量实时进行归一化处理。
    def __init__(self, power=2):# 参数power：指定归一化的 “范数类型”，即 L-p 范数中的p，222（如特征向量归一化到单位球面上）
        super(Normalize, self).__init__()
        self.power = power

    def execute(self, x):  #  核心计算方法（execute）：执行归一化逻辑
        #Jittor 框架中用execute替代 PyTorch 的forward作为模块的前向传播入口，接收输入张量x，返回归一化后的张量out。
        # 步骤1：计算输入张量x的L-p范数（沿通道维度，保留维度）
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        # 执行归一化（out的计算）
        out = x / (norm + 1e-7)
        return out

#对输入的 2D 特征图（如卷积层输出）进行全局最大池化与L-2 范数归一化，最终输出紧凑的全局特征向量。它本质是一个 “特征压缩与标准化模块”，常用于特征提取、对比学习或相似度计算等场景
class PoolingF(nn.Module):
    def __init__(self):
        super(PoolingF, self).__init__()
        model = [nn.AdaptiveMaxPool2d(1)]  # 单个自适应最大池化层
        self.model = nn.Sequential(*model)# 用Sequential封装（便于后续扩展，如添加卷积/激活层）
        self.l2norm = Normalize(2)

    def execute(self, x):  
        return self.l2norm(self.model(x))

#核心功能是对2D特征图进行“自适应平均池化-维度重排-展平-L2归一化”的组合操作
# 本质就是特征维度重构与标准化模块“，常用于需要将空间特征拆解为独立样本的场景
# 与上一个不同，关键在于对空间特征进行”拆解“，不是将每个样本的空间特征压缩为 1 个全局向量，而是将每个样本的空间区域拆分为多个独立的局部向量，再统一标准化
class ReshapeF(nn.Module):
    def __init__(self):
        super(ReshapeF, self).__init__()
        model = [nn.AdaptiveAvgPool2d(4)]  # # 输出固定为4×4的自适应平均池化
        # ：nn.AdaptiveAvgPool2d(4)会将任意空间尺寸的输入特征图，压缩为固定的4×4空间尺寸（即输出空间维度为H=4, W=4）。
        # 池化逻辑：对输入特征图的空间区域做 “平均池化”（而非PoolingF的最大池化），每个4×4输出对应输入的一个局部区域的平均值，保留局部区域的整体信息，避免最大池化可能丢失的全局趋势。
        self.model = nn.Sequential(*model)
        self.l2norm = Normalize(2)

    def execute(self, x):  
        # 步骤1：自适应平均池化，将输入压缩为4×4空间尺寸
        x = self.model(x)
        # 步骤2：维度重排（permute）+ 展平（flatten），重构为独立向量
        x_reshape = x.permute(0, 2, 3, 1).flatten(0, 2)  
        # 步骤3：L-2归一化，标准化特征尺度
        return self.l2norm(x_reshape)


class StridedConvF(nn.Module):
    def __init__(self, init_type='normal', init_gain=0.02, gpu_ids=[]):
        super().__init__()
        self.l2_norm = Normalize(2)
        self.mlps = {}
        self.moving_averages = {}
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, x):
        C, H = x.shape[1], x.shape[2]  # 替换size为shape
        n_down = int(np.rint(np.log2(H / 32)))
        mlp = []
        for i in range(n_down):
            mlp.append(nn.Conv2d(C, max(C // 2, 64), 3, stride=2))  # Jittor的Conv2d
            mlp.append(nn.ReLU())
            C = max(C // 2, 64)
        mlp.append(nn.Conv2d(C, 64, 3))
        mlp = nn.Sequential(*mlp)
        init_net(mlp, self.init_type, self.init_gain, self.gpu_ids)
        return mlp

    def update_moving_average(self, key, x):
        if key not in self.moving_averages:
            self.moving_averages[key] = x.detach()  # Jittor支持detach

        self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001

    def execute(self, x, use_instance_norm=False):  # 替换forward为execute
        C, H = x.shape[1], x.shape[2]  # 替换size为shape
        key = '%d_%d' % (C, H)
        if key not in self.mlps:
            self.mlps[key] = self.create_mlp(x)
            self.add_module("child_%s" % key, self.mlps[key])  # Jittor支持add_module
        mlp = self.mlps[key]
        x = mlp(x)
        self.update_moving_average(key, x)
        x = x - self.moving_averages[key]
        if use_instance_norm:
            x = nn.instance_norm(x)  # 替换F.instance_norm为nn.instance_norm
        return self.l2_norm(x)


class PatchSampleF(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
            # Jittor自动管理设备，无需显式cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def execute(self, feats, num_patches=64, patch_ids=None):  # 替换forward为execute
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]  # size→shape
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # 替换torch.randperm为np.random.permutation
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]
                # 替换torch.tensor为jt.array，无需显式指定device（Jittor自动管理）
                patch_id = jt.array(patch_id, dtype=jt.int64)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape→flatten
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                # view→reshape，size→shape
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids


class G_Resnet(nn.Module):
    def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64,
                 norm=None, nl_layer=None):
        super(G_Resnet, self).__init__()
        n_downsample = num_downs
        pad_type = 'reflect'
        self.enc_content = ContentEncoder(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
        if nz == 0:
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)
        else:
            self.dec = Decoder_all(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer, pad_type=pad_type, nz=nz)

    def decode(self, content, style=None):
        return self.dec(content, style)

    def execute(self, image, style=None, nce_layers=[], encode_only=False):  # 替换forward为execute
        content, feats = self.enc_content(image, nce_layers=nce_layers, encode_only=encode_only)
        if encode_only:
            return feats
        else:
            images_recon = self.decode(content, style)
            if len(nce_layers) > 0:
                return images_recon, feats
            else:
                return images_recon


##################################################################################
# Encoder and Decoders
##################################################################################

# 提取“风格特征”：将输入图像映射为风格特征向量
class E_adaIN(nn.Module):# “adaIN” 代表与 “Adaptive Instance Normalization（自适应实例归一化）” 相关
    def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4,
                 norm=None, nl_layer=None, vae=False):
        super(E_adaIN, self).__init__()
        self.enc_style = StyleEncoder(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)
    # norm：归一化层类型（此处固定为'none'，表示风格编码器不使用归一化层，避免破坏风格特征）；

    def execute(self, image):  # 替换forward为execute
        style = self.enc_style(image)
        return style
# 输入image：待提取风格的图像张量（形状通常为[batch_size, input_nc, height, width]）；
#处理过程：将图像传入self.enc_style（风格编码器），得到风格特征向量style；
#输出style：提取的风格特征（形状为[batch_size, output_nc]或根据vae参数可能为(均值, 方差)）。



#它能将高分辨率图像（如[batch, 3, 256, 256]）通过多轮卷积下采样，逐步提取图像的风格信息（如色彩分布、纹理模式、笔触风格等），最终输出紧凑的风格特征向量。

#专门用于提取图像风格特征的核心编码器网络。它通过 “卷积下采样 + 全局平均池化” 的经典结构，将输入图像压缩为低维风格特征向量，同时支持变分自编码器（VAE）模式以输出特征的均值和方差。
class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
        super(StyleEncoder, self).__init__()
        self.vae = vae
        self.model = []
        # 初始卷积层（保持空间尺寸，提取基础风格特征），组件：Conv2dBlock（自定义卷积块，包含 “卷积 + 归一化 + 激活”）；
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        # 前 2 轮下采样（通道数翻倍，空间尺寸减半）
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
        self.model += [nn.AdaptiveAvgPool2d(1)]  
        if self.vae:
            # VAE模式：用全连接层输出均值和方差（各为style_dim维度）
            self.fc_mean = nn.Linear(dim, style_dim)
            self.fc_var = nn.Linear(dim, style_dim)
        else:
            # 普通模式：用1×1卷积将通道数压缩为style_dim（直接输出风格向量）
            self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def execute(self, x):  
        if self.vae:
            output = self.model(x)  
            output = output.reshape(x.shape[0], -1) 
            output_mean = self.fc_mean(output)
            output_var = self.fc_var(output)
            return output_mean, output_var
        else:
            return self.model(x).reshape(x.shape[0], -1) 

# 内容编码：初始卷积 + 多轮下采样 + 残差块
class ContentEncoder(nn.Module):
    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
            dim *= 2
        # 作用：通过下采样压缩空间冗余，同时翻倍通道数以提取更抽象的内容特征（如物体部件组合）。
        # 残差块（增强特征表达，缓解梯度问题），每个残差块保持通道数dim和空间尺寸不变。
        # 组件：ResBlocks（多个残差块的组合，每个残差块包含 “卷积 + 归一化 + 激活 + 跳跃连接”）；

        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim


    def execute(self, x, nce_layers=[], encode_only=False): 
        if len(nce_layers) > 0:
            # 模式1：需要提取指定中间层的特征（用于NCE损失等计算）
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in nce_layers:
                    feats.append(feat)
                if layer_id == nce_layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            return self.model(x), None

# 解码器，将低维的压缩特征逐步上采样并融合，最中恢复为高分辨率的目标图像【从特征到图像的重建任务】
class Decoder_all(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder_all, self).__init__()
        # AdaIN 残差融合块（self.resnet_block）
        # nz参数控制是否融合额外特征（如风格特征）—— 若nz>0，残差块会先将输入特征与额外特征（如风格向量y）拼接（通过cat_feature函数），再进行残差变换；
        self.resnet_block = ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
        self.n_blocks = 0
        # 多轮上采样块（动态创建block_0~block_{n_upsample-1}）
        for i in range(n_upsample):
            # 上采样块结构：上采样（尺寸翻倍）→ 卷积（通道数减半）
            block = [Upsample2(scale_factor=2),# 上采样层（如双线性插值，将尺寸×2）
                      Conv2dBlock(
                        dim + nz,  # 输入通道：主特征通道数 + 额外特征通道数（nz）
                        dim // 2,  # 输出通道：通道数减半（与编码器下采样时通道翻倍对应）
                        5, 1, 2,   # 5×1卷积核、步长1、填充2（确保上采样后尺寸不变）
                        norm='ln', # 归一化：层归一化（LayerNorm），适合上采样的细节保持
                        activation=activ, 
                        pad_type='reflect'
                        )
                    ]
            # 动态创建子模块（如block_0、block_1），避免列表存储，便于后续调用
            setattr(self, 'block_{:d}'.format(self.n_blocks), nn.Sequential(*block))
            self.n_blocks += 1
            dim //= 2# 每轮上采样后，通道数减半（与编码器下采样的通道翻倍形成对称）
        # use reflection padding in the last conv layer
        # 动态创建最后一个卷积块，输出目标图像
        # 无归一化：避免影响图像像素的最终分布
        # 激活函数：tanh（输出像素值映射到[-1,1]，符合图像数据范围）
        setattr(self, 'block_{:d}'.format(self.n_blocks), Conv2dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
        self.n_blocks += 1
    # 特征融合与图像生成，接收主输入特征x（如内容特征）和可选的额外融合特征y（如风格特征），输出最终生成的图像
    def execute(self, x, y=None):
        if y is not None:
            # 步骤1：残差块融合主特征x与额外特征y
            output = self.resnet_block(cat_feature(x, y))
            # 步骤2：多轮上采样块处理（逐步恢复尺寸，融合y）
            for n in range(self.n_blocks):
                block = getattr(self, 'block_{:d}'.format(n)) # 动态获取上采样块/最终卷积块
                if n > 0:
                    # 上采样块（n≥1）：先拼接output与y，再处理
                    output = block(cat_feature(output, y))
                else:
                    # 第一个上采样块（n=0）：直接处理残差块输出（已融合y）
                    output = block(output)
            return output

# 将低维特征映射回高分辨率图像的解码器网络
# 它通过 “残差块特征增强 + 多轮上采样 + 最终卷积” 的流程，实现从压缩特征到图像的重建，同时支持额外特征（如风格特征）的融合
# 与之前的Decoder_all相比，它采用更简洁的nn.Sequential线性结构，适用于特征融合逻辑相对简单的场景
# Decoder用nn.Sequential按顺序组织所有层，结构更简洁；而Decoder_all通过动态创建子模块实现更灵活的特征融合。两者核心目标一致，但Decoder在实现上更轻量化。
class Decoder(nn.Module):
    def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
        super(Decoder, self).__init__()

        self.model = []
        # 第一步：AdaIN 残差块（特征增强与融合准备）
        self.model += [ResBlocks(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
        # upsampling blocks
        for i in range(n_upsample):
            # 确定上采样块的输入通道数（首次上采样需包含额外特征nz，后续无需）
            if i == 0:
                input_dim = dim + nz # 第1轮上采样：主特征通道 + 额外特征通道
            else:
                input_dim = dim # 后续上采样：仅主特征通道
            self.model += [Upsample2(scale_factor=2), Conv2dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
            dim //= 2
        # use reflection padding in the last conv layer
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
        self.model = nn.Sequential(*self.model)

    def execute(self, x, y=None): 
        if y is not None:
            return self.model(cat_feature(x, y))
        else:
            return self.model(x)


##################################################################################
# Sequential Models
##################################################################################

# 多个残差块（ResBlock）的串联组合
# 通过将单个残差块重复堆叠，构建深层残差网络结构，核心作用是增强特征表达能力、
# 缓解深层网络的梯度消失 / 爆炸问题，同时支持额外特征（如风格特征）的融合
# norm	归一化层类型（如'inst'实例归一化、'batch'批归一化，稳定训练）
# pad_type	卷积层的填充方式（如'reflect'反射填充、'zero'零填充，保护边缘细节）
class ResBlocks(nn.Module):
    # dim	输入 / 输出特征的通道数（残差块保持通道数不变，确保残差连接可直接相加）
    # num_blocks	残差块的堆叠数量（如num_blocks=4表示串联 4 个相同结构的ResBlock）
    def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            # 为每个残差块传入相同的结构参数（通道数、归一化、激活等）
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
        self.model = nn.Sequential(*self.model) # 封装为顺序执行的网络

    def execute(self, x):  # 将输入特征x传入串联的残差块集合，输出增强后的特征
        return self.model(x)


##################################################################################
# Basic Blocks
##################################################################################
# 特征拼接
def cat_feature(x, y):
    # expand→broadcast，size→shape
    # 通道数为两者之和，空间尺寸与x一致
    # 将低维特征y（通常是 1×1 空间尺寸的向量，如风格特征）通过广播机制扩展到与主特征x相同的空间尺寸，然后在通道维度上拼接接两者，形成融合特征。
    y_expand = y.reshape(y.shape[0], y.shape[1], 1, 1).broadcast(
        y.shape[0], y.shape[1], x.shape[2], x.shape[3])
    x_cat = jt.concat([x, y_expand], 1)  # torch.cat→jt.concat
    return x_cat


class ResBlock(nn.Module):
    def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
        super(ResBlock, self).__init__()

        model = []
        # 第一层卷积：融合额外特征，输出dim通道
        # dim + nz,  # 输入通道：主特征通道数 + 额外特征通道数（nz）
        # dim,       # 输出通道：主特征通道数（dim）
        # 3, 1, 1,   # 3×3卷积核、步长1、填充1（确保空间尺寸不变）
        model += [Conv2dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        # 第二层卷积：特征变换，输出dim+nz通道（与输入通道一致，便于残差相加）
        # activation='none'# 最后一层无激活，避免抑制残差相加后的特征
        # 仅第一层卷积使用激活函数，第二层不使用（避免激活函数对残差相加的结果产生过度抑制）。
        model += [Conv2dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)# 封装为顺序执行的卷积块

    def execute(self, x): 
        residual = x # 保存输入特征作为残差（跳过连接）
        out = self.model(x) # 输入特征经过卷积块变换
        out += residual # 残差相加：变换后的特征 + 原始输入特征
        return out

# 卷积功能块
# 填充（Padding）→卷积（Conv2d）→归一化（Normalization）→激活（Activation）
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # 填充层（self.pad）—— 保护空间尺寸与边缘特征
        # 在卷积前对特征图边缘进行像素补充，避免卷积后空间尺寸缩小，同时减少边缘特征的信息损失
        if pad_type == 'reflect':
            # 反射填充：以特征图边缘为对称轴，反射补充像素（适合风格迁移，避免边缘模糊）
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            # 零填充：用0补充边缘像素（通用填充方式，简单高效）
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
#示例：若输入特征图尺寸为64×64，使用 3×3 卷积核、padding=1：
#填充后尺寸变为66×66，卷积后尺寸仍为64×64（无尺寸损失）；
#反射填充会复制边缘像素的对称值（如边缘像素为 [1,2,3]，填充后为 [2,1,2,3,2]），零填充则补充 0。
        
        # 归一化层（self.norm）—— 稳定训练，加速收敛
        # 归一化的核心作用是：将卷积后的特征图像素值 “标准化”（如均值接近 0、方差接近 1），避免梯度消失或爆炸，加速深层网络训练
        norm_dim = output_dim
        if norm == 'batch':
            # 批归一化（BatchNorm）：对批次内所有样本的同一通道做归一化（适合大批次训练）
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'inst':
            # 实例归一化（InstanceNorm）：对单个样本的单个通道做归一化（适合风格迁移、生成任务）
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'ln':
            # 层归一化（LayerNorm）：对单个样本的所有通道做归一化（适合小批次或上采样场景）
            self.norm = nn.LayerNorm(norm_dim)  # 假设LayerNorm已适配Jittor
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # 激活函数（self.activation）—— 引入非线性，增强表达
        # 对归一化后的特征图施加非线性变换，让网络能学习复杂的非线性关系（如边缘、纹理、结构等）
        if activation == 'relu':
            self.activation = nn.ReLU() # ReLU：简单高效，缓解梯度消失（输出非负）
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)# LeakyReLU：保留负区间小梯度，避免神经元死亡
        elif activation == 'prelu':
            self.activation = nn.PReLU()# PReLU：自适应学习负区间斜率，更灵活
        elif activation == 'selu':
            self.activation = nn.SELU()# SELU：自归一化，无需额外归一化层
        elif activation == 'tanh':
            self.activation = nn.Tanh()# Tanh：输出映射到[-1,1]，适合图像生成（像素值范围匹配）
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        self.use_bias = True # 是否使用偏置（默认True，归一化层通常不依赖偏置，但此处简化设置）
        # 卷积层（self.conv）—— 核心特征变换
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def execute(self, x):  # 固定顺序的特征变换
        # 步骤1：填充（保护尺寸/边缘）→ 卷积（提取特征+通道转换）
        x = self.conv(self.pad(x))
        # 步骤2：归一化（稳定训练）（若配置了归一化层）
        if self.norm:
            x = self.norm(x)
        # 步骤3：激活（引入  非线性）（若配置了激活函数）
        if self.activation:
            x = self.activation(x)
        return x

# 全连接（Linear）→归一化（Normalization）→激活（Activation），标准化一维向量特征的变换流程
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # 初始化全连接层， 核心向量变换，将输入的一维向量通过线性变换映射到目标维度
        # 公式 out = x · W + b（W为权重矩阵，b为偏置项）。
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
        # 全连接层会将输入的 “样本 × 输入维度” 矩阵，与 “输入维度 × 输出维度” 的权重矩阵相乘，得到 “样本 × 输出维度” 的结果；
        # 偏置项b为每个输出维度添加一个常数偏移，增强线性变换的灵活性。

        # 初始化归一化层
        # 将全连接层输出的向量 “标准化”（如均值接近 0、方差接近 1），避免一维特征值过大 / 过小导致的梯度消失或爆炸，加速深层网络训练
        norm_dim = output_dim # 归一化层的作用维度
        if norm == 'batch':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'inst':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        #上面这个是1D的，在前面那个是2D的，需要了解一下他的区别

        # 初始化激活函数（Jittor无需inplace参数）
        # 激活函数的核心作用是：对归一化后的向量施加非线性变换，让网络能学习复杂的非线性关系（如从特征向量到类别概率的映射）
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def execute(self, x):  # 替换forward为execute
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:        # 特征维度数量（如输入为[batch, 64, 32, 32]，则特征维度为 64）

            out = self.activation(out)
        return out


##################################################################################
# 归一化层，层归一化模块，与批归一化（BatchNorm）不同，它独立处理每个样本的特征，通过标准化特征值来稳定网络训练，尤其适合小批量数据或序列数据场景
##################################################################################
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features # 记录特征维度
        self.affine = affine # 是否启用仿射变换
        self.eps = eps # 数值稳定参数

        if self.affine:
            # 初始化可学习参数（gamma初始为均匀分布，beta初始为0）
            # 替换torch.Tensor为jt.array，nn.Parameter用法一致
            self.gamma = nn.Parameter(jt.array(num_features).uniform_())# 缩放参数（形状：[num_features]）
            self.beta = nn.Parameter(jt.zeros(num_features))# 平移参数（形状：[num_features]）


    def execute(self, x):  # 替换forward为execute
        # 步骤1：计算单个样本的特征均值（mean）和方差（std）
        # 构建reshape的目标形状：保留批次维度，将其他维度展平（如[batch, C, H, W]→[batch, C*H*W]）
        shape = [-1] + [1] * (x.dim() - 1)
        # 替换view为reshape，size为shape
        mean = x.reshape(x.shape[0], -1).mean(1).reshape(*shape)
        std = x.reshape(x.shape[0], -1).std(1).reshape(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.reshape(*shape) + self.beta.reshape(*shape)
        return x


class PatchDomainClassifier(nn.Module):
    def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256, gpu_ids=[]):
        super(PatchDomainClassifier, self).__init__()
        input_dims = [3, 128, 256, 256, 256]
        self.use_mlp = use_mlp
        self.nc = nc  # 硬编码
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

        for mlp_id, dim in enumerate(input_dims):
            mlp = nn.Sequential(*[nn.Linear(dim, self.nc), nn.ReLU(), nn.Linear(self.nc, 1)])
            setattr(self, 'mlp_%d' % mlp_id, mlp)
            # Jittor自动管理设备，无需显式cuda()
        networks.init_net(self, self.init_type, self.init_gain, self.gpu_ids)  # 使用之前转换的init_net

    def execute(self, feat, mlp_id):  # 替换forward为execute
        mlp = getattr(self, 'mlp_%d' % mlp_id)
        return mlp(feat)


class AdaIN(nn.Module):
    def __init__(self, num_features, style_dim=1):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
        self.domain_variable = None

    def execute(self, x):  # 替换forward为execute
        assert self.domain_variable is not None, 'Please assign domain variable to AdaIN!'
        h = self.fc(self.domain_variable)
        h = h.reshape(h.shape[0], h.shape[1], 1, 1)  # view→reshape
        gamma, beta = jt.chunk(h, chunks=2, dim=1)  # torch.chunk→jt.chunk
        return (1 + gamma) * self.norm(x) + beta


class MultiAdaINResnetGenerator(nn.Module):
    """Resnet-based generator with multi AdaIN layers"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        assert(n_blocks >= 0)
        super(MultiAdaINResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):  # 添加下采样层
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU()]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(),
                          networks.Downsample(ngf * mult * 2)]  # 使用转换后的Downsample

        mult = 2 ** n_downsampling
        for i in range(n_blocks // 2):  # 添加ResNet块
            model += [networks.ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]  # 使用转换后的ResnetBlock

        self.encoder = nn.Sequential(*model)

        # 风格编码器A
        dim = 64
        style_encoder = [nn.Conv2d(input_nc, dim, 7, 1, 3), nn.ReLU()]
        for i in range(2):
            style_encoder += [nn.Conv2d(dim, dim * 2, 4, 2, 1), nn.ReLU()]
            dim *= 2
        for i in range(2):
            style_encoder += [nn.Conv2d(dim, dim, 4, 2, 1), nn.ReLU()]
        style_encoder += [nn.AdaptiveAvgPool2d(1)]
        style_encoder += [nn.Conv2d(dim, opt.style_dim, 1, 1, 0)]
        self.style_encoder_A = nn.Sequential(*style_encoder)

        # 风格编码器B
        dim = 64
        style_encoder = [nn.Conv2d(input_nc, dim, 7, 1, 3), nn.ReLU()]
        for i in range(2):
            style_encoder += [nn.Conv2d(dim, dim * 2, 4, 2, 1), nn.ReLU()]
            dim *= 2
        for i in range(2):
            style_encoder += [nn.Conv2d(dim, dim, 4, 2, 1), nn.ReLU()]
        style_encoder += [nn.AdaptiveAvgPool2d(1)]
        style_encoder += [nn.Conv2d(dim, opt.style_dim, 1, 1, 0)]
        self.style_encoder_B = nn.Sequential(*style_encoder)

        # 解码器
        model = []
        norm_layer = functools.partial(AdaIN, style_dim=opt.style_dim)
        for i in range(n_blocks - n_blocks // 2):  # 添加ResNet块
            model += [networks.ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]  # 使用转换后的ResnetBlock

        for i in range(n_downsampling):  # 添加上采样层
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU()]
            else:
                model += [networks.Upsample(ngf * mult),  # 使用转换后的Upsample
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU()]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.decoder = nn.Sequential(*model)
        self.domain_embedding = nn.Embedding(2, opt.style_dim)  # Jittor支持nn.Embedding
        self.style_dim = opt.style_dim

    def assign_domain_variable(self, domain_variable, model):
        # 移除device相关操作，Jittor自动管理
        if len(domain_variable.shape) == 1:  # size→shape
            domain_variable = domain_variable.unsqueeze(-1)
        assert len(domain_variable.shape) == 2
        for m in model.modules():
            if m.__class__.__name__ == "AdaIN":
                m.domain_variable = domain_variable

    def execute(self, input, layers=[], mode='encode'):  # 替换forward为execute
        if mode == 'encode':
            return self.encoder(input)
        elif mode == 'encode_style_A':
            return self.style_encoder_A(input) + 1.
        elif mode == 'encode_style_B':
            return self.style_encoder_B(input) - 1.
        elif mode == 'decode':
            latent, domain_variable = input
            self.assign_domain_variable(domain_variable, self.decoder)
            return self.decoder(latent)
        elif mode == 'decode_and_extract':
            latent, domain_variable = input
            self.assign_domain_variable(domain_variable, self.decoder)
            if len(layers) > 0:
                feat = latent
                feats = []
                for layer_id, layer in enumerate(self.decoder):
                    feat = layer(feat)
                    if layer_id in layers:
                        feats.append(feat)
                    if layer_id == layers[-1]:
                        return feats  # 仅返回中间特征
        elif mode == 'extract':
            if len(layers) > 0:
                feat = input
                feats = []
                for layer_id, layer in enumerate(self.encoder):
                    feat = layer(feat)
                    if layer_id in layers:
                        feats.append(feat)
                    if layer_id == layers[-1]:
                        return feats  # 仅返回中间特征
                if layers[-1] > layer_id:
                    # 替换torch.zeros为jt.zeros，移除to(device)
                    self.assign_domain_variable(jt.zeros([len(input)]), self.decoder)
                    for _, layer in enumerate(self.decoder):
                        layer_id += 1
                        feat = layer(feat)
                        if layer_id in layers:
                            feats.append(feat)
                        if layer_id == layers[-1]:
                            return feats  # 仅返回中间特征
                return feats  # 返回输出和中间特征






import jittor as jt
import jittor.nn as nn
import functools
from . import networks  # 假设包含转换后的Downsample、Upsample、ResnetBlock等


class AdaINResnetGenerator(jt.nn.Module):
    """Resnet-based generator with AdaIN layers (适配 Jittor)"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=jt.nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=6, padding_type='reflect', 
                 no_antialias=False, no_antialias_up=False, opt=None):
        assert(n_blocks >= 0)
        super(AdaINResnetGenerator, self).__init__()
        self.opt = opt
        
        # 判断是否使用偏置（Jittor 中 InstanceNorm 通常不需要偏置）
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func == jt.nn.InstanceNorm2d
        else:
            use_bias = norm_layer == jt.nn.InstanceNorm2d

        # 编码器部分
        model = [
            jt.nn.ReflectionPad2d(3),
            jt.nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            jt.nn.ReLU()  # Jittor 无 inplace 参数，移除
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # 下采样层
            mult = 2 ** i
            if no_antialias:
                model += [
                    jt.nn.Conv2d(ngf * mult, ngf * mult * 2, 
                                kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    jt.nn.ReLU()
                ]
            else:
                model += [
                    jt.nn.Conv2d(ngf * mult, ngf * mult * 2, 
                                kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    jt.nn.ReLU(),
                    networks.Downsample(ngf * mult * 2)  # 确保 Downsample 已适配 Jittor
                ]

        mult = 2 ** n_downsampling
        # 添加 ResNet 块（前半部分）
        for i in range(n_blocks // 2):
            model += [
                networks.ResnetBlock(  # 确保 ResnetBlock 已适配 Jittor
                    ngf * mult, 
                    padding_type=padding_type, 
                    norm_layer=norm_layer, 
                    use_dropout=use_dropout, 
                    use_bias=use_bias
                )
            ]

        self.encoder = jt.nn.Sequential(*model)
        
        # 解码器部分
        model = []
        # 替换为 Jittor 版本的 AdaIN 层
        norm_layer = functools.partial(networks.AdaIN, style_dim=opt.style_dim)
        
        # 添加 ResNet 块（后半部分）
        for i in range(n_blocks - n_blocks // 2):
            model += [
                networks.ResnetBlock(
                    ngf * mult, 
                    padding_type=padding_type, 
                    norm_layer=norm_layer, 
                    use_dropout=use_dropout, 
                    use_bias=use_bias
                )
            ]

        # 上采样层
        for i in range(n_downsampling):
            mult = 2 **(n_downsampling - i)
            if no_antialias_up:
                model += [
                    jt.nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    jt.nn.ReLU()
                ]
            else:
                model += [
                    networks.Upsample(ngf * mult),  # 确保 Upsample 已适配 Jittor
                    jt.nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                              kernel_size=3, stride=1,
                              padding=1,
                              bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    jt.nn.ReLU()
                ]
        
        # 输出层
        model += [
            jt.nn.ReflectionPad2d(3),
            jt.nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            jt.nn.Tanh()
        ]

        self.decoder = jt.nn.Sequential(*model)
        # Jittor 嵌入层（替代 PyTorch 的 nn.Embedding）
        self.domain_embedding = jt.nn.Embedding(2, opt.style_dim)
        self.style_dim = opt.style_dim

    def assign_domain_variable(self, domain_variable, model):
        # 适配 Jittor 张量操作，移除设备相关代码（Jittor 自动管理设备）
        if len(domain_variable.shape) == 1:
            domain_variable = domain_variable.unsqueeze(-1)  # 替换 .unsqueeze(1) 为 .unsqueeze(-1)
        assert len(domain_variable.shape) == 2, f"domain_variable 维度错误，当前维度: {len(domain_variable.shape)}"
        
        if self.style_dim > 1:
            # 用 Jittor 张量创建函数替代 PyTorch 的 torch.zeros/torch.ones
            start = self.domain_embedding(jt.zeros([len(domain_variable)], dtype=jt.int64))
            end = self.domain_embedding(jt.ones([len(domain_variable)], dtype=jt.int64))
            domain_variable = start + domain_variable * (end - start)
        
        # 为 AdaIN 层分配 domain_variable
        for m in model.modules():
            if m.__class__.__name__ == "AdaIN":  # 确保 AdaIN 类名正确
                m.domain_variable = domain_variable

    def execute(self, input, layers=[], mode='encode'):  # Jittor 用 execute 替代 forward
        if mode == 'encode':
            return self.encoder(input)
        
        elif mode == 'decode':
            latent, domain_variable = input
            self.assign_domain_variable(domain_variable, self.decoder)
            return self.decoder(latent)
        
        elif mode == 'decode_and_extract':
            latent, domain_variable = input
            self.assign_domain_variable(domain_variable, self.decoder)
            if len(layers) > 0:
                feat = latent
                feats = []
                for layer_id, layer in enumerate(self.decoder):
                    feat = layer(feat)
                    if layer_id in layers:
                        feats.append(feat)
                    if layer_id == layers[-1]:
                        return feats  # 返回指定层的中间特征
        
        elif mode == 'extract':
            if len(layers) > 0:
                feat = input
                feats = []
                # 从编码器提取特征
                for layer_id, layer in enumerate(self.encoder):
                    feat = layer(feat)
                    if layer_id in layers:
                        feats.append(feat)
                    if layer_id == layers[-1]:
                        return feats
                
                # 若需从解码器继续提取特征
                if layers[-1] > len(self.encoder) - 1:
                    # 初始化 domain_variable 为全 0（适配 Jittor 张量）
                    self.assign_domain_variable(jt.zeros([len(input), 1]), self.decoder)
                    # 遍历解码器层
                    for layer in self.decoder:
                        layer_id += 1  # 延续编码器的 layer_id 计数
                        feat = layer(feat)
                        if layer_id in layers:
                            feats.append(feat)
                        if layer_id == layers[-1]:
                            return feats
                return feats




class ResnetGenerator(nn.Module):
    """Resnet-based generator with standard ResNet blocks"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()]  # 移除inplace=True

        n_downsampling = 2
        for i in range(n_downsampling):  # 添加下采样层
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU()]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(),
                          networks.Downsample(ngf * mult * 2)]  # 使用转换后的Downsample

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # 添加ResNet块
            model += [networks.ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]  # 使用转换后的ResnetBlock

        for i in range(n_downsampling):  # 添加上采样层
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU()]
            else:
                model += [networks.Upsample(ngf * mult),  # 使用转换后的Upsample
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU()]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def execute(self, input, layers=[], encode_only=False):  # forward→execute
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return feats  # 仅返回中间特征
            return feat, feats  # 返回输出和中间特征
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake











class ResnetDecoder(nn.Module):
    """Resnet-based decoder with Resnet blocks and upsampling operations"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        assert(n_blocks >= 0)
        super(ResnetDecoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []
        n_downsampling = 2
        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # 添加ResNet块
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # 添加-upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU()]  # 移除inplace=True
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU()]  # 移除inplace=True

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def execute(self, input):  # forward→execute
        """Standard forward"""
        return self.model(input)


class ResnetEncoder(nn.Module):
    """Resnet-based encoder with downsampling and Resnet blocks"""

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False):
        assert(n_blocks >= 0)
        super(ResnetEncoder, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU()]  # 移除inplace=True

        n_downsampling = 2
        for i in range(n_downsampling):  # 添加下采样层
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU()]  # 移除inplace=True
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(),  # 移除inplace=True
                          Downsample(ngf * mult * 2)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # 添加ResNet块
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def execute(self, input):  # forward→execute
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block with skip connections"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block"""
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                      norm_layer(dim), 
                      nn.ReLU()]  # 移除inplace=True
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                      norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def execute(self, x):  # forward→execute
        """Forward with skip connections"""
        out = x + self.conv_block(x)  # 添加跳跃连接
        return out


class UnetGenerator(nn.Module):
    """Unet-based generator with skip connections"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        # 递归构建U-Net结构
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # 最内层
        for i in range(num_downs - 5):  # 中间层
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # 从ngf*8逐步减少到ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # 最外层

    def execute(self, input):  # forward→execute
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Unet子模块带跳跃连接: X → 下采样 → 子模块 → 上采样 → 拼接(X, 输出)"""

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2)  # 移除inplace=True
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU()  # 移除inplace=True
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def execute(self, x):  # forward→execute
        if self.outermost:
            return self.model(x)
        else:  # 添加跳跃连接
            return jt.concat([x, self.model(x)], 1)  # torch.cat→jt.concat


import jittor as jt
from jittor import nn
import functools

# 步骤1：实现Jittor版本的谱归一化函数
def spectral_norm(module, name='weight', n_power_iterations=1):
    """Jittor谱归一化，用于稳定GAN训练"""
    weight = getattr(module, name)
    dim = weight.shape[0]
    u = jt.randn(dim, requires_grad=False)
    
    # 幂迭代近似谱范数
    for _ in range(n_power_iterations):
        v = jt.normalize(jt.sum(weight * u.unsqueeze(1).unsqueeze(2).unsqueeze(3), dim=0))
        u = jt.normalize(jt.sum(weight.transpose(0,1) * v.unsqueeze(1).unsqueeze(2).unsqueeze(3), dim=0))
    
    sigma = jt.sum(weight * u.unsqueeze(1).unsqueeze(2).unsqueeze(3), dim=0) * v
    sigma = sigma.max()
    setattr(module, name, weight / sigma)
    return module

# 步骤2：定义下采样层（原代码中的Downsample，假设为平均池化下采样）
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 2x2平均池化下采样
    
    def execute(self, x):
        return self.pool(x)

# 步骤3：转换GlobalImageDis类
class GlobalImageDis(nn.Module):
    # 多尺度判别器结构
    def __init__(self, input_dim=3):
        super(GlobalImageDis, self).__init__()
        self.n_layer = 4
        self.gan_type = 'lsgan'
        self.dim = 64
        self.norm = 'none'
        self.activ = 'lrelu'
        self.scales = [5]  # 尺度参数
        self.pad_type = 'reflect'
        self.input_dim = input_dim
        self.cnns = nn.ModuleList()  # Jittor的ModuleList
        for scale in self.scales:
            self.cnns.append(self._make_net(scale))

    def _make_net(self, num_layers):
        dim = self.dim
        cnn_x = []
        # 用自定义spectral_norm包装卷积层，替换原SN
        cnn_x += [spectral_norm(nn.Conv2d(self.input_dim, dim, 4, 2, 1)), nn.LeakyReLU(0.2)]
        for i in range(num_layers):
            outdim = min(512, dim*2)
            cnn_x += [spectral_norm(nn.Conv2d(dim, outdim, 4, 2, 1)), nn.LeakyReLU(0.2)]
            dim = outdim
        cnn_x += [spectral_norm(nn.Conv2d(dim, 1, 4, 1, 0))]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    # Jittor用execute作为前向传播方法，替换PyTorch的forward
    def execute(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
        return outputs[0]

# 步骤4：转换GlobalDiscriminator类
class GlobalDiscriminator(nn.Module):
    """PatchGAN判别器"""

    def __init__(self, input_nc, ndf=64, n_layers=5, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super(GlobalDiscriminator, self).__init__()
        # 判断是否需要偏置（与归一化层相关）
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if no_antialias:
            # 无抗锯齿下采样：直接用stride=2的卷积
            sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), 
                       nn.LeakyReLU(0.2, True)]
        else:
            # 有抗锯齿下采样：卷积+激活+下采样层
            sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw)), 
                       nn.LeakyReLU(0.2, True), 
                       Downsample(ndf)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 逐步增加滤波器数量
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                                          kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)
                ]

        # 输出1通道的预测图
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=0))]
        self.model = nn.Sequential(*sequence)

    # 前向传播用execute
    def execute(self, input):
        return self.model(input)

class NLayerDiscriminator(nn.Module):
    """定义PatchGAN判别器"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """构建PatchGAN判别器
        
        参数:
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 最后一个卷积层的滤波器数量
            n_layers (int)  -- 判别器中的卷积层数量
            norm_layer      -- 归一化层
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # BatchNorm2d有仿射参数，不需要偏置
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if no_antialias:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), 
                       nn.LeakyReLU(0.2), 
                       Downsample(ndf)]
            
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # 逐渐增加滤波器数量
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # 输出1通道预测图
        self.model = nn.Sequential(*sequence)

    def execute(self, input):
        """标准前向传播"""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """定义1x1 PatchGAN判别器 (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """构建1x1 PatchGAN判别器
        
        参数:
            input_nc (int)  -- 输入图像的通道数
            ndf (int)       -- 最后一个卷积层的滤波器数量
            norm_layer      -- 归一化层
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # BatchNorm2d有仿射参数，不需要需要偏置
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def execute(self, input):
        """标准前向传播"""
        return self.net(input)

class PatchDiscriminator(NLayerDiscriminator):
    """定义PatchGAN判别器"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)

    def execute(self, input):
        B, C, H, W = input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        size = 16
        Y = H // size
        X = W // size
        input = input.view(B, C, Y, size, X, size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
        return super().execute(input)


class GroupedChannelNorm(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups

    def execute(self, x):
        shape = list(x.shape)
        new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
        x = x.view(*new_shape)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x_norm = (x - mean) / (std + 1e-7)
        return x_norm.view(*shape)
