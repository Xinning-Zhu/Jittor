import jittor as jt
from .cut_model import CUTModel  


class SinCUTModel(CUTModel):
    """ 此类实现单图像翻译模型
    源自论文: Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = CUTModel.modify_commandline_options(parser, is_train)
        parser.add_argument('--lambda_R1', type=float, default=1.0,
                            help='weight for the R1 gradient penalty')
        parser.add_argument('--lambda_identity', type=float, default=1.0,
                            help='the "identity preservation loss"')

        # 设置默认参数
        parser.set_defaults(
            nce_includes_all_negatives_from_minibatch=True,
            dataset_mode="singleimage",
            netG="stylegan2",
            stylegan2_G_num_downsampling=1,
            netD="stylegan2",
            gan_mode="nonsaturating",
            num_patches=1,
            nce_layers="0,2,4",
            lambda_NCE=4.0,
            ngf=10,
            ndf=8,
            lr=0.002,
            beta1=0.0,
            beta2=0.99,
            load_size=1024,
            crop_size=64,
            preprocess="zoom_and_patch",
        )

        if is_train:
            parser.set_defaults(
                preprocess="zoom_and_patch",
                batch_size=16,
                save_epoch_freq=1,
                save_latest_freq=20000,
                n_epochs=8,
                n_epochs_decay=8,
            )
        else:
            parser.set_defaults(
                preprocess="none", 
                batch_size=1,
                num_test=1,
            )
            
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        if self.isTrain:
            # 根据配置添加损失名称
            if opt.lambda_R1 > 0.0:
                self.loss_names += ['D_R1']
            if opt.lambda_identity > 0.0:
                self.loss_names += ['idt']

    def compute_D_loss(self):
        # Jittor中通过设置requires_grad=True开启梯度计算
        self.real_B.requires_grad = True
        GAN_loss_D = super().compute_D_loss()
        # 计算R1损失
        self.loss_D_R1 = self.R1_loss(self.pred_real, self.real_B)
        # 总判别器损失
        self.loss_D = GAN_loss_D + self.loss_D_R1
        return self.loss_D

    def compute_G_loss(self):
        CUT_loss_G = super().compute_G_loss()
        # 计算身份保留损失（L1损失）
        self.loss_idt = jt.nn.l1_loss(self.idt_B, self.real_B) * self.opt.lambda_identity
        return CUT_loss_G + self.loss_idt

    def R1_loss(self, real_pred, real_img):
        # Jittor中使用grad函数计算梯度，create_graph=True保留计算图
        grad_real = jt.grad(real_pred.sum(), real_img, create_graph=True)
        # 计算梯度惩罚
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty * (self.opt.lambda_R1 * 0.5)
