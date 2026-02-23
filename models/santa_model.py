import numpy as np
import jittor as jt
from .base_model import BaseModel  
from . import networks  # 自定义网络结构（生成器，判别器）
from .patchnce import PatchNCELoss  # 对比学习损失函数，用于增强特征匹配
import util.util as util  
import os
from jittor import nn
from util.image_pool import ImagePool  # 图像池，用于存储生成的假图像，稳定GAN训练

# 用于非配对图像到图像的翻译任务（如风格迁移、域转换等），继承自 CUT 和 FastCUT 模型的核心思想(对比学习)
class SANTAModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # 添加模型特有的超参数
        # 各类损失权重，平衡不同损失对模型训练的影响
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='weight for reconstruction loss')
        parser.add_argument('--lambda_idt', type=float, default=5.0, help='weight for identity loss')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss')
        parser.add_argument('--lambda_path', type=float, default=0.01, help='weight for path loss')
        # 指定计算路径损失的网络层，控制特征平滑性约束的层级
        parser.add_argument('--path_layers', type=str, default='0,3,6,10,14', help='compute path loss on which layers')
        # 定义生成器编码得到的 “风格潜在向量” 的维度，维度越大，可编码的风格细节越丰富，但计算成本和过拟合风险也会增加
        parser.add_argument('--style_dim', type=int, default=8, help='style dimension')
        # 控制路径损失计算中 “相邻域标签的采样间隔范围”
        parser.add_argument('--path_interval_min', type=float, default=0.05, help='minimum interval for path sampling')
        parser.add_argument('--path_interval_max', type=float, default=0.10, help='maximum interval for path sampling')
        # 添加噪声标准差，用于正则化
        parser.add_argument('--noise_std', type=float, default=1.0, help='standard deviation for noise')
        # 实验标签，用于区分不同训练任务的输出路径
        parser.add_argument('--tag', type=str, default='debug', help='experiment tag')
        # no_html=True：禁用 HTML 可视化报告生成，减少训练过程中的磁盘 IO 开销
        # pool_size=0：设置生成图像池的大小为 0，即不缓存历史生成图像。图像池的作用是通过采样历史假图像稳定判别器训练，此处禁用可能简化训练流程，但可能降低稳定性
        parser.set_defaults(no_html=True, pool_size=0)  
        # 自动生成唯一的模型标识（model_id）,并将其设置为模型的名称（name）
        opt, _ = parser.parse_known_args()
        if opt.phase != 'test':
            model_id = f'{opt.tag}'
            model_id += '/' + os.path.basename(opt.dataroot.strip('/')) + f'_{opt.direction}'
            model_id += f'/lam{opt.lambda_path}_layers{opt.path_layers}_dim{opt.style_dim}_rec{opt.lambda_rec}_idt{opt.lambda_idt}_pool{opt.pool_size}_noise{opt.noise_std}_kl{opt.lambda_kl}'
            parser.set_defaults(name=model_id)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # 定义损失名称和可视化图像名称
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G_rec', 'G_idt', 'G_kl', 'G_path', 'd1', 'd2']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # 动态添加每层的能量损失
        self.path_layers = [int(i) for i in self.opt.path_layers.split(',')]
        for l in self.path_layers:
            self.loss_names += [f'energy_{l}']
        
        # 根据训练 / 测试模式定义需要创建和保存的网络
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  
            self.model_names = ['G']

        # 生成器
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, 
            opt.normG, not opt.no_dropout, opt.init_type, 
            opt.init_gain, opt.no_antialias, opt.no_antialias_up, 
            self.gpu_ids, opt
        )
        self.d_A = jt.zeros([1]) # 源域标签
        self.d_B = jt.ones([1]) # 目标域标签

        # 判别器，只有训练时创建
        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, 
                opt.normD, opt.init_type, opt.init_gain, 
                opt.no_antialias, self.gpu_ids, opt
            )
            self.fake_B_pool = ImagePool(opt.pool_size)

            # 损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode)
            self.criterionNCE = []
            # L1损失，用于重构损失和身份损失
            self.criterionIdt = nn.L1Loss()
            # 优化器
            self.optimizer_G = jt.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = jt.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    # 训练参数优化
    def optimize_parameters(self):
        # 前向传播，生成假图像
        self.forward()

        # 判别器D优化
        self.set_requires_grad(self.netD, True)  # 启用判别器梯度
        self.optimizer_D.zero_grad()  # 清空梯度
        self.loss_D = self.compute_D_loss()  # 计算判别器损失
        self.optimizer_D.backward(self.loss_D)  # 反向传播计算梯度
        self.optimizer_D.step()  # 更新参数

        # 生成器G优化
        self.set_requires_grad(self.netD, False)  # 控制网络是否计算梯度，避免训练 G 时更新 D 的参数
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.optimizer_G.backward(self.loss_G)  
        self.optimizer_G.step()

    # 输入数据处理
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'  # 转换方向
        self.real_A = input['A' if AtoB else 'B']  # 源域图像
        self.real_B = input['B' if AtoB else 'A']  # 目标域图像
        self.image_paths = input['A_paths' if AtoB else 'B_paths']  # 源域图像路径
        
    # 前向传播，生成器的核心功能实现，分为 "编码" 和 "解码" 两步
    def forward(self):
        real = jt.concat([self.real_A, self.real_B], dim=0)  # 拼接源域和目标域图像
        latents = self.netG(real, mode='encode')  # 编码为潜在向量
        if self.isTrain and self.opt.noise_std > 0:  # 训练时添加噪声，正则化
            noise = jt.normal(0, self.opt.noise_std, size=latents.shape)
            self.mu = latents 
            latents = latents + noise
        # 拆分源域和目标域的潜在向量
        self.latent_A, self.latent_B = latents.chunk(2, dim=0)
        # 生成三类图像：重构A、生成B、身份B
        ds = jt.concat([self.d_A, self.d_B, self.d_B], 0).unsqueeze(-1)  # 域标签（A或B）
        latents = jt.concat([self.latent_A, self.latent_A, self.latent_B], 0)  # 拼接潜在向量
        images = self.netG((latents, ds), mode='decode')  # 解码生成图像
        self.rec_A, self.fake_B, self.idt_B = images.chunk(3, dim=0)  

    # 使用@jt.no_grad()装饰器禁用梯度计算（节省内存且加速计算）
    @jt.no_grad()
    def single_forward(self):
        latent = self.netG(self.real_A, mode='encode')
        out = self.netG((latent, self.d_B), mode='decode')
        
    # 判别器损失计算，区分真实图像（real_B）和生成图像（fake_B）
    def compute_D_loss(self):
        # 从图像池中采样假图像
        fake = self.fake_B_pool.query(self.fake_B.detach())
        # 判别器对假图像的预测（应判定为假）
        pred_fake = self.netD(fake)
        # 计算损失：希望判别器将假图像判定为“假”
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # 总损失为两者的平均值
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    # 生成器损失计算，生成高质量的目标域图像，同时满足多重约束
    def compute_G_loss(self):
        # GAN损失：生成图像应被判别器判定为真
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True).mean()
        # 重构损失：源域重构图像应与原图一致
        self.loss_G_rec = self.criterionIdt(self.rec_A, self.real_A).mean()
        # 身份损失：目标域身份映射应与原图一致
        self.loss_G_idt = self.criterionIdt(self.idt_B, self.real_B).mean()
        self.loss_d1 = jt.array(0.)
        self.loss_d2 = jt.array(0.)
        for l in self.path_layers:
            setattr(self, f'loss_energy_{l}', 0)
        # KL损失：约束潜在向量的噪声
        if self.opt.noise_std > 0:
            self.loss_G_kl = (self.mu ** 2).mean()
        else:
            self.loss_G_kl = 0
        self.loss_n_dots = 0
        # 路径损失：约束潜在空间的平滑性
        if self.opt.lambda_path > 0:
            self.loss_G_path = self.compute_path_losses()
        else:
            self.loss_G_path = jt.array(0.)

        # 总损失：加权求和
        self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN + \
                      self.opt.lambda_rec * self.loss_G_rec + \
                      self.opt.lambda_idt * self.loss_G_idt + \
                      self.opt.lambda_kl * self.loss_G_kl + \
                      self.opt.lambda_path * self.loss_G_path
        return self.loss_G


    # 路径损失计算
    def compute_path_losses(self):
        #  随机生成中心值d1_center（[0,1)区间均匀分布）
        d1_center = jt.rand(len(self.latent_A))  
        # 随机生成间隔interval（在[path_interval_min, path_interval_max]范围内）
        interval_rand = jt.rand(len(self.latent_A))  
        interval = self.opt.path_interval_min + interval_rand * (self.opt.path_interval_max - self.opt.path_interval_min)
    
        d1 = (d1_center + interval).clamp(0, 1)  # 域标签1
        d2 = (d1_center - interval).clamp(0, 1)  # 域标签2（与d1接近）
        # 拼接源域潜在向量
        latents = jt.concat([self.latent_A, self.latent_A], 0)
        # 拼接域标签d1和d2
        ds = jt.concat([d1, d2], 0).unsqueeze(-1)
        # 生成对应d1和d2的特征
        features = self.netG((latents, ds), layers=self.path_layers, mode='decode_and_extract')

        self.loss_d1 = d1.mean()
        self.loss_d2 = d2.mean()

        # 计算特征随域标签变化的平滑性（雅可比矩阵的平方和）
        loss_path = 0  
        for id, feats in enumerate(features):
            x_d1, x_d2 = feats.chunk(2, dim=0)
            jacobian = (x_d1 - x_d2) / jt.maximum(d1 - d2, jt.ones_like(d1) * 0.1)  # 近似特征变化率
            energy = (jacobian ** 2).mean()  # 能量（约束变化平滑性）
            setattr(self, f'loss_energy_{self.path_layers[id]}', energy.item()) # 记录每层能量损失
            loss_path += energy  # 累加各层损失
        # 取平均作为总路径损失
        loss_path = loss_path / len(features)
        return loss_path


    @jt.no_grad()
    # 评估潜在空间的整体平滑性
    def compute_whole_path_length(self):
        small_int = 0.1  # 步长间隔
        # 生成从0到1-small_int的均匀采样点（共10个点：0, 0.1, 0.2, ..., 0.9）
        linsp = np.linspace(0, 1 - small_int, int(1 / small_int))
        losses = [0 for _ in range(5)]  # 存储各层的路径损失
    
        for d2 in linsp:
            d1 = d2 + small_int  # d1 = d2 + 0.1（相邻两个点）
            d1 = jt.array([d1], dtype=jt.float32)
            d2 = jt.array([d2], dtype=jt.float32)

            # 生成并提取对应点对的中间特征
            latents = jt.concat([self.latent_A, self.latent_A], 0)
            ds = jt.concat([d1, d2], 0).unsqueeze(-1) 
            features = self.netG((latents, ds), layers=self.path_layers, mode='decode_and_extract')
            
            loss_path = 0
            for id, feats in enumerate(features):
                # 拆分d1和d2对应的特征
                x_d1, x_d2 = feats.chunk(2, dim=0)
                # 计算特征随域标签变化的梯度（近似雅可比矩阵）
                jacobian = (x_d1 - x_d2) / jt.maximum(d1 - d2, jt.ones_like(d1) * 0.1)
                # 能量：梯度的平方和
                energy = (jacobian ** 2).mean()
                loss_path += energy
                # 累加各层损失（除以10是因为有10个间隔）
                losses[id] += energy.item() / 10
        return losses

    @jt.no_grad()
    # 生成中间过渡图像
    def interpolation(self, x_a, x_b):
        self.netG.eval()  # 生成器切换到评估模式（关闭 dropout 等）
        # 根据转换方向确定源域图像（A→B则用x_a，B→A则用x_b）
        if self.opt.direction == 'AtoB':
            x = x_a
        else:
            x = x_b
        interps = []  # 存储插值结果
        # 对前8张图像进行插值（避免数量过多）
        for i in range(min(x.shape[0], 8)):
            # 编码源域图像得到潜在向量
            h_a = self.netG(x[i].unsqueeze(0), mode='encode')
            d = 0.2  # 从0.2开始插值
            local_interps = []
            # 生成d=0.2, 0.4, 0.6, 0.8对应的中间图像
            while d < 1.:
                d_t = jt.array([d]).unsqueeze(-1)  # 域标签d
                # 用潜在向量h_a和域标签d生成中间图像
                local_interps.append(self.netG((h_a, d_t), mode='decode'))
                d += 0.2  # 步长0.2
            # 拼接中间图像并添加到结果列表
            local_interps = jt.concat(local_interps, 0)
            interps.append(local_interps)
        self.netG.train()  # 切回训练模式
        return interps
    
    # 实际场景中的图像转换，批量处理测试集图像，用于定量评估模型性能
    @jt.no_grad()
    def translate(self, x):
        # 生成器切换到评估模式
        self.netG.eval() 
        # 编码输入图像得到潜在向量
        h = self.netG(x, mode='encode')
        # 用潜在向量和目标域标签（d_B）解码生成目标域图像
        out = self.netG((h, self.d_B), mode='decode')
        self.netG.train()  # 切回训练模式
        return out

    # 生成多组对比图像，评估模型重建和转换能力
    @jt.no_grad()
    def sample(self, x_a, x_b):
        self.netG.eval()
        # 根据转换方向调整源域和目标域（确保x_a为源域，x_b为目标域）
        if self.opt.direction == 'BtoA':
            x_a, x_b = x_b, x_a
        x_a_recon, x_b_recon, x_ab = [], [], []
        # 对每张图像生成三组结果
        for i in range(x_a.shape[0]):
            # 编码源域和目标域图像
            h_a = self.netG(x_a[i].unsqueeze(0), mode='encode')  # 源域潜在向量
            h_b = self.netG(x_b[i].unsqueeze(0), mode='encode')  # 目标域潜在向量
            # 1. 源域重建：用源域潜在向量和源域标签（d_A）解码
            x_a_recon.append(self.netG((h_a, self.d_A), mode='decode'))
            # 2. 目标域重建：用目标域潜在向量和目标域标签（d_B）解码
            x_b_recon.append(self.netG((h_b, self.d_B), mode='decode'))
            # 3. 跨域转换：用源域潜在向量和目标域标签（d_B）解码
            x_ab.append(self.netG((h_a, self.d_B), mode='decode'))
        # 拼接结果
        x_a_recon, x_b_recon = jt.concat(x_a_recon), jt.concat(x_b_recon)
        x_ab = jt.concat(x_ab)
        self.netG.train()
        # 返回：原图A、A的重建、A转B的结果、原图B、B的重建
        return x_a, x_a_recon, x_ab, x_b, x_b_recon