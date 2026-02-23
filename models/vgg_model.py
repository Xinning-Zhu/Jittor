import jittor as jt
from .base_model import BaseModel
from . import networks
import os
from util.image_pool import ImagePool
import jittor.nn as nn
from jittor.models import vgg16  

# 预处理输入图像
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        mean = jt.array([0.485, 0.456, 0.406])
        std = jt.array([0.229, 0.224, 0.225])
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def execute(self, img):
        # 归一化图像
        return (img - self.mean) / self.std

# 提取图像在不同层级的特征，用于计算感知损失（Perceptual Loss），确保生成图像与目标图像在语义层面一致
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 加载预训练VGG16特征层
        features = vgg16(pretrained=True).features
        self.relu1_1 = jt.nn.Sequential()
        self.relu1_2 = jt.nn.Sequential()

        self.relu2_1 = jt.nn.Sequential()
        self.relu2_2 = jt.nn.Sequential()

        self.relu3_1 = jt.nn.Sequential()
        self.relu3_2 = jt.nn.Sequential()
        self.relu3_3 = jt.nn.Sequential()

        self.relu4_1 = jt.nn.Sequential()
        self.relu4_2 = jt.nn.Sequential()
        self.relu4_3 = jt.nn.Sequential()

        self.relu5_1 = jt.nn.Sequential()
        self.relu5_2 = jt.nn.Sequential()
        self.relu5_3 = jt.nn.Sequential()

        # 构建各层特征提取器
        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(23, 26):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(28, 30):
            self.relu5_3.add_module(str(x), features[x])

    def execute(self, x, layers=None, encode_only=False, resize=False):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)

        relu4_1 = self.relu4_1(relu3_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)

        relu5_1 = self.relu5_1(relu4_3)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,
            'relu2_1': relu2_1,
            'relu2_2': relu2_2,
            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
        }
        if encode_only:
            if len(layers) > 0:
                feats = []
                for layer, key in enumerate(out):
                    if layer in layers:
                        feats.append(out[key])
                return feats
            else:
                return out['relu3_1']
        return out


class VGGModel(BaseModel):
    """ 此类实现CUT和FastCUT模型，源自论文
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """ 配置CUT模型特定的选项
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='GAN损失权重：GAN(G(X))')
        parser.add_argument('--lambda_rec', type=float, default=5.0, help='重建损失权重')
        parser.add_argument('--lambda_idt', type=float, default=5.0, help='身份损失权重')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='KL损失权重')
        parser.add_argument('--lambda_path', type=float, default=0.01, help='路径损失权重')
        parser.add_argument('--path_layers', type=str, default='0,3,6,10,14', help='计算NCE损失的层')
        parser.add_argument('--style_dim', type=int, default=8, help='风格维度')
        parser.add_argument('--path_interval_min', type=float, default=0.05, help='路径间隔最小值')
        parser.add_argument('--path_interval_max', type=float, default=0.10, help='路径间隔最大值')
        parser.add_argument('--noise_std', type=float, default=1.0, help='噪声标准差')
        parser.add_argument('--tag', type=str, default='debug', help='实验标签')
        parser.set_defaults(no_html=True, pool_size=0)  
        opt, _ = parser.parse_known_args()
        model_id = f'{opt.tag}'
        model_id += '/'+os.path.basename(opt.dataroot.strip('/')) + f'_{opt.direction}'
        model_id += f'/lam{opt.lambda_path}_layers{opt.path_layers}_dim{opt.style_dim}_rec{opt.lambda_rec}_idt{opt.lambda_idt}_pool{opt.pool_size}_noise{opt.noise_std}_kl{opt.lambda_kl}'

        parser.set_defaults(name=model_id)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G_rec', 'G_idt', 'G_kl', 'G_path',  'd1', 'd2']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.path_layers = [int(i) for i in self.opt.path_layers.split(',')]
        for l in self.path_layers:
            self.loss_names += [f'energy_{l}']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else: 
            self.model_names = ['G']

        # 定义网络（生成器和判别器）
        self.netG = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, 
            not opt.no_dropout, opt.init_type, opt.init_gain, 
            opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt
        )
        self.d_A = jt.zeros([1]) 
        self.d_B = jt.ones([1])
        print(self.netG)

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt
            )
            self.fake_B_pool = ImagePool(opt.pool_size)  # 创建图像缓冲区存储生成的图像
            print(self.netD)
            self.netPre = VGG16() 
            self.normalization = Normalization()

            # 定义损失函数
            self.criterionGAN = networks.GANLoss(opt.gan_mode) 
            self.criterionNCE = []

            self.criterionIdt = jt.nn.L1Loss()
            # 定义优化器
            self.optimizer_G = jt.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = jt.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def optimize_parameters(self):
        # 前向传播
        self.execute()

        # 更新判别器D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # 更新生成器G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """从数据加载器解包输入数据并执行必要的预处理
        参数:
            input (dict): 包含数据本身及其元数据
        选项'direction'用于交换A域和B域
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B']  
        self.real_B = input['B' if AtoB else 'A']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def execute(self):
        """运行前向传播；被<optimize_parameters>和<test>调用"""
        real = jt.cat([self.real_A, self.real_B], dim=0)  
        latents = self.netG(real, mode='encode')
        if self.isTrain and self.opt.noise_std > 0:
            noise = jt.normal(mean=0, std=self.opt.noise_std, shape=latents.shape)
            self.mu = latents
            latents = latents + noise
        self.latent_A, self.latent_B = latents.chunk(2, dim=0) 
        ds = jt.cat([self.d_A, self.d_B, self.d_B], 0).unsqueeze(-1)
        latents = jt.cat([self.latent_A, self.latent_A, self.latent_B], 0)
        images = self.netG((latents, ds), mode='decode')
        self.rec_A, self.fake_B, self.idt_B = images.chunk(3, dim=0)

    def compute_D_loss(self):
        """计算判别器的GAN损失"""
        fake = self.fake_B_pool.query(self.fake_B.detach()) 
        # 假样本；通过detach停止梯度传播到生成器
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # 真样本
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # 合并损失并计算梯度
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """计算生成器的GAN和NCE损失"""
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True).mean()
        self.loss_G_rec = self.criterionIdt(self.rec_A, self.real_A).mean()
        self.loss_G_idt = self.criterionIdt(self.idt_B, self.real_B).mean()
        self.loss_d1 = jt.float32(0.) 
        self.loss_d2 = jt.float32(0.)
        for l in self.path_layers:
            setattr(self, f'loss_energy_{l}', 0)
        if self.opt.noise_std > 0:
            self.loss_G_kl = jt.pow(self.mu, 2).mean()
        else:
            self.loss_G_kl = 0
        self.loss_n_dots = 0
        if self.opt.lambda_path > 0:
            self.loss_G_path = self.compute_path_losses()
        else:
            self.loss_G_path = jt.float32(0.)
        self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN +\
                      self.opt.lambda_rec * self.loss_G_rec + \
                      self.opt.lambda_idt * self.loss_G_idt + \
                      self.opt.lambda_kl * self.loss_G_kl + \
                      self.opt.lambda_path * self.loss_G_path
        return self.loss_G

    def compute_path_losses(self):
        norm_rec_A = self.normalization((self.rec_A + 1) * 0.5)
        norm_fake_B = self.normalization((self.fake_B + 1) * 0.5)
        feats_src = self.netPre(norm_rec_A, [4, 7, 9], encode_only=True)
        feats_tgt = self.netPre(norm_fake_B, [4, 7, 9], encode_only=True)
        total_loss = 0.0
        for i, (feat_src, feat_tgt) in enumerate(zip(feats_src, feats_tgt)):
            loss = (feat_src - feat_tgt) **2
            total_loss += loss.mean()
        return total_loss

    @jt.no_grad() 
    def interpolation(self, x_a, x_b):
        self.netG.eval()
        if self.opt.direction == 'AtoB':
            x = x_a
        else:
            x = x_b
        interps = []
        for i in range(min(x.shape[0], 8)):
            h_a = self.netG(x[i].unsqueeze(0), mode='encode')
            d = 0.0
            local_interps = []
            local_interps.append(x[i].unsqueeze(0))
            while d <= 1.:
                d_t = jt.array([d]).unsqueeze(-1)  
                local_interps.append(self.netG((h_a, d_t), mode='decode'))
                d += 0.1
            local_interps = jt.cat(local_interps, 0) 
            interps.append(local_interps)
        self.netG.train()
        return interps

    @jt.no_grad()
    def translate(self, x):
        self.netG.eval()
        h = self.netG(x, mode='encode')
        out = self.netG((h, self.d_B), mode='decode')
        self.netG.train()
        return out

    @jt.no_grad()
    def sample(self, x_a, x_b):
        self.netG.eval()
        if self.opt.direction == 'BtoA':
            x_a, x_b = x_b, x_a
        x_a_recon, x_b_recon, x_ab = [], [], []
        for i in range(x_a.shape[0]):
            h_a = self.netG(x_a[i].unsqueeze(0), mode='encode')
            h_b = self.netG(x_b[i].unsqueeze(0), mode='encode')
            x_a_recon.append(self.netG((h_a, self.d_A), mode='decode'))
            x_b_recon.append(self.netG((h_b, self.d_B), mode='decode'))
            x_ab.append(self.netG((h_a, self.d_B), mode='decode'))
        x_a_recon, x_b_recon = jt.cat(x_a_recon), jt.cat(x_b_recon)  
        x_ab = jt.cat(x_ab)
        self.netG.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon