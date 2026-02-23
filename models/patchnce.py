from packaging import version
import jittor as jt
from jittor import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        # 初始化交叉熵损失函数
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        # 定义掩码数据类型（mask_dtype），用于后续过滤无效样本对
        self.mask_dtype = jt.bool

    # 正向计算
    def execute(self, feat_q, feat_k):  
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.stop_grad()

        # 正样本计算
        l_pos = jt.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)
        )
        l_pos = l_pos.view(num_patches, 1)

        # 负样本计算
        if self.opt.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = jt.bmm(feat_q, feat_k.transpose(2, 1))

        # 掩码掉对角元素
        diagonal = jt.eye(npatches, dtype=self.mask_dtype)[None, :, :]  
        l_neg_curbatch = l_neg_curbatch.masked_fill(diagonal, -10.0)  
        l_neg = l_neg_curbatch.view(-1, npatches)

        # 拼接正负样本并归一化
        out = jt.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        # 计算交叉熵损失，标签为全0（正样本对应索引0）
        loss = self.cross_entropy_loss(
            out, 
            jt.zeros(out.size(0), dtype=jt.int64) )

        return loss