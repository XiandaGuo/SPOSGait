import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple
from torchvision.ops import RoIAlign


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, c_in, s, h_in, w_in]
            Out x: [n, c_out, s, h_out, w_out]
        """
        n, c, s, h, w = x.size()
        x = self.forward_block(x.transpose(
            1, 2).view(-1, c, h, w), *args, **kwargs)
        output_size = x.size()
        return x.reshape(n, s, *output_size[1:]).transpose(1, 2).contiguous()


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, dim=2, options={}):
        """
            In  seqs: [n, c, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **options)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **options))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)

class PackSequenceWrapper_bb(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper_bb, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, seq_dim=1, **kwargs):
        """
            In  seqs: [n, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **kwargs)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
            rets.append(self.pooling_func(narrowed_seq, **kwargs))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [n, c_in, p]
            out: [n, c_out, p]
        """
        x = x.permute(2, 0, 1).contiguous()
        # print('x', x.size(), self.fc_bin.size())   # x torch.Size([64, 8, 256]) torch.Size([64, 256, 256])
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out.permute(1, 2, 0).contiguous()


class SeparateBNNecks(nn.Module):
    """
        GaitSet: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if self.norm:
            feature = F.normalize(feature, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            logits = feature.matmul(self.fc_bin)
        return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


class GaitAlign(nn.Module):
    def __init__(self, H=64, W=44, eps=1, **kwargs):
        super(GaitAlign, self).__init__()
        self.H, self.W, self.eps = H, W, eps
        self.Pad = nn.ZeroPad2d((int(self.W / 2), int(self.W / 2), 0, 0))
        self.RoiPool = RoIAlign((self.H, self.W), 1, sampling_ratio=-1)

    def forward(self, feature_map, binary_mask, w_h_ratio):
        """
           In  sils:         [n, c, h, w]
               w_h_ratio:    [n, 1]
           Out aligned_sils: [n, c, H, W]
        """
        n, c, h, w = feature_map.size()
        # w_h_ratio = w_h_ratio.repeat(1, 1) # [n, 1]
        w_h_ratio = w_h_ratio.view(-1, 1)  # [n, 1]

        h_sum = binary_mask.sum(-1)  # [n, c, h]
        _ = (h_sum >= self.eps).float().cumsum(axis=-1)  # [n, c, h]
        h_top = (_ == 0).float().sum(-1)  # [n, c]
        h_bot = (_ != torch.max(_, dim=-1, keepdim=True)
                 [0]).float().sum(-1) + 1.  # [n, c]

        w_sum = binary_mask.sum(-2)  # [n, c, w]
        w_cumsum = w_sum.cumsum(axis=-1)  # [n, c, w]
        w_h_sum = w_sum.sum(-1).unsqueeze(-1)  # [n, c, 1]
        w_center = (w_cumsum < w_h_sum / 2.).float().sum(-1)  # [n, c]

        p1 = self.W - self.H * w_h_ratio
        p1 = p1 / 2.
        p1 = torch.clamp(p1, min=0)  # [n, c]
        t_w = w_h_ratio * self.H / w
        p2 = p1 / t_w  # [n, c]

        height = h_bot - h_top  # [n, c]
        width = height * w / h  # [n, c]
        width_p = int(self.W / 2)

        feature_map = self.Pad(feature_map)
        w_center = w_center + width_p  # [n, c]

        w_left = w_center - width / 2 - p2  # [n, c]
        w_right = w_center + width / 2 + p2  # [n, c]

        w_left = torch.clamp(w_left, min=0., max=w+2*width_p)
        w_right = torch.clamp(w_right, min=0., max=w+2*width_p)

        boxes = torch.cat([w_left, h_top, w_right, h_bot], dim=-1)
        # index of bbox in batch
        box_index = torch.arange(n, device=feature_map.device)
        rois = torch.cat([box_index.view(-1, 1), boxes], -1)
        crops = self.RoiPool(feature_map, rois)  # [n, c, H, W]
        return crops


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False

### CSTL ###
class BasicConv2d_forCSTL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d_forCSTL, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv2d(x)
        return F.leaky_relu(x, inplace=True)

class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h, w)


class Attention(nn.Module):
    def __init__(self, in_dims, part_num, num_head):
        super(Attention, self).__init__()
        self.part_num = part_num
        self.num_head = num_head
        self.dim_head = in_dims // num_head

        self.scale = self.dim_head ** (-0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.to_qkv = nn.Conv1d(in_dims * part_num, in_dims * 3 * part_num, 1, bias=False, groups=part_num)
        self.to_out = nn.Conv1d(in_dims * part_num, in_dims * part_num, 1, bias=False, groups=part_num)

    def forward(self, x):
        n, p, c, d = x.shape

        qkv = self.to_qkv(x.view(n, p * c, d))
        qkv = qkv.view(n, p, 3, self.num_head, self.dim_head, d).permute(2, 0, 3, 1, 4, 5).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # [n, num_head, p, dim_head, d]

        dots = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = self.softmax(dots)

        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)  # [n, num_head, p, dim_head, d]

        out = self.to_out(out.permute(0, 2, 1, 3, 4).contiguous().view(n, -1, d)).view(n, p, c, d)

        return out, attn


class PreNorm(nn.Module):
    def __init__(self, part_num, in_dims, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(in_dims)
        self.fn = fn

    def forward(self, x):
        n, p, c, d = x.shape

        return self.fn(self.norm(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous())


class FeedForward(nn.Module):
    def __init__(self, in_dims, part_num, decay=16):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dims * part_num, in_dims * part_num // decay, 1, bias=False, groups=part_num),
            nn.BatchNorm1d(in_dims * part_num // decay),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_dims * part_num // decay, in_dims * part_num, 1, bias=False, groups=part_num)
        )

    def forward(self, x):
        n, p, c, d = x.size()
        out = self.net(x.view(n, -1, d)).view(n, p, c, d)
        return out


class Transformer(nn.Module):
    def __init__(self, in_dims, depth, num_head, decay, part_num):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(part_num, in_dims, Attention(in_dims, part_num, num_head)),
                PreNorm(part_num, in_dims, FeedForward(in_dims, part_num, decay))
            ]))

    def forward(self, x):  # nxpxcxd
        for attn, ff in self.layers:
            x = attn(x)[0] + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, in_dims, out_dims, part_num, depth, num_head, decay, kernel_size=1, stride=1):
        super(ViT, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.stride = stride

        self.raw_embedding = nn.Conv1d(in_dims * part_num, out_dims * part_num, 1, bias=False, groups=part_num)
        self.proj = nn.Conv1d(in_dims, out_dims, kernel_size, stride, padding=kernel_size // 2, bias=False,
                              groups=in_dims)

        self.transformer = Transformer(in_dims, depth, num_head, decay, part_num)

        self.activate = nn.LeakyReLU(inplace=True)

    def seq_embedding(self, x):
        n, p, c, d = x.size()
        seq_embedded = self.raw_embedding(x.view(n, -1, d)).view(n, p, -1, d)

        return seq_embedded

    def pos_embedding(self, x):
        n, p, c, d = x.size()
        feat_token = self.proj(x.view(-1, c, d)).view(n, p, -1, d)
        if self.stride == 1:
            feat_token += x
        return feat_token

    def forward(self, x):
        embedded_feature = self.seq_embedding(x) + self.pos_embedding(x)
        trans_feature = self.transformer(embedded_feature)
        return self.activate(trans_feature)


import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
import seaborn as sns
from matplotlib import pyplot as plt


def conv1d(in_planes, out_planes, kernel_size, has_bias=False, **kwargs):
    return nn.Conv1d(in_planes, out_planes, kernel_size, bias=has_bias, **kwargs)


def mlp_sigmoid(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, in_planes // 16, kernel_size, **kwargs),
                         nn.BatchNorm1d(in_planes // 16),
                         nn.LeakyReLU(inplace=True),
                         conv1d(in_planes // 16, out_planes, kernel_size, **kwargs),
                         nn.Sigmoid())


def conv_bn(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, out_planes, kernel_size, **kwargs),
                         nn.BatchNorm1d(out_planes))


def plot_heatmap(corr_matrix):
    n, h, p, s, s, = corr_matrix.shape

    topk_part_index = torch.topk(corr_matrix.sum(-2), 3, dim=-1, largest=True)[1]
    print("topk_part_index: ", topk_part_index, "size: ", topk_part_index.shape)
    print(topk_part_index.squeeze(0)[0][15])

    corr_matrix = corr_matrix.squeeze(0)[0]
    print(corr_matrix.shape)

    sns.set_theme()
    f, ax = plt.subplots(figsize=(16, 16))
    matrix = corr_matrix[15, :, :].cpu().numpy()
    out = sns.heatmap(matrix, ax=ax)
    plt.show()

    return


class MSTE(nn.Module):
    def __init__(self, in_planes, out_planes, part_num):
        super(MSTE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.part_num = part_num

        self.score = mlp_sigmoid(in_planes * part_num, in_planes * part_num, 1, groups=part_num)

        self.short_term = nn.ModuleList(
            [conv_bn(in_planes * part_num, out_planes * part_num, 3, padding=1, groups=part_num),
             conv_bn(in_planes * part_num, out_planes * part_num, 3, padding=1, groups=part_num)])

    def get_frame_level(self, x):
        return x

    def get_short_term(self, x):
        n, p, c, s = x.size()
        temp = self.short_term[0](x.view(n, -1, s))
        short_term_feature = temp + self.short_term[1](temp)

        return short_term_feature.view(n, p, c, s)

    def get_long_term(self, x):
        n, p, c, s = x.size()
        pred_score = self.score(x.view(n, -1, s)).view(n, p, c, s)
        long_term_feature = x.mul(pred_score).sum(-1).div(pred_score.sum(-1))
        long_term_feature = long_term_feature.unsqueeze(3).repeat(1, 1, 1, s)

        return long_term_feature

    def forward(self, x):
        return self.get_frame_level(x), self.get_short_term(x), self.get_long_term(x)


class ATA(nn.Module):
    def __init__(self, in_planes=256, out_planes=256, groups=32, depth=1, heads=4, decay=16, kernel_size=3, stride=1):
        super(ATA, self).__init__()
        self.groups = groups
        self.t = ViT(in_planes, out_planes, groups, depth, heads, decay, kernel_size, stride)
        self.decay_c = conv_bn(in_planes * groups * 3, in_planes * groups, 1, groups=groups)
        self.activate = nn.LeakyReLU(inplace=True)

    def forward(self, t_f, t_s, t_l):
        n, p, c, s = t_f.size()
        temporal_feature = self.decay_c(torch.cat([t_f, t_s, t_l], 2).view(n, -1, s))
        temporal_feature = self.t(temporal_feature.view(n, p, -1, s)) + temporal_feature.view(n, p, -1, s)
        weighted_feature = self.activate(temporal_feature.max(-1)[0]).permute(1, 0, 2).contiguous()

        return weighted_feature


class SpatialLearning(nn.Module):
    def __init__(self, in_planes, out_planes, num_head, part_num, class_num, topk_num):
        super(SpatialLearning, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.part_num = part_num
        self.class_num = class_num
        self.num_head = num_head
        self.dim_head = in_planes // num_head
        self.topk_num = topk_num

        self.decay_c = nn.Conv1d(in_planes * part_num * 3, in_planes * part_num, 1, bias=False, groups=part_num)

        self.frame_corr = Attention(in_planes, part_num, num_head)

        # self.to_qkv = nn.Conv1d(in_planes * part_num, in_planes * part_num * 3, 1, bias=False, groups=part_num)
        self.scale = self.dim_head ** (-0.5)
        self.softmax = nn.Softmax(dim=-1)
        # self.to_out = nn.Conv1d(in_planes * part_num, out_planes * part_num, 1, bias=False, groups=part_num)

        self.bn = nn.ModuleList()
        for i in range(part_num):
            self.bn.append(nn.BatchNorm1d(in_planes))

        self.fc = nn.Parameter(
            init.xavier_uniform_(
                torch.zeros(1, in_planes, class_num)))

    def frame_attention(self, t_f):  # frame-level
        n, p, c, s = t_f.size()

        qkv = self.to_qkv(t_f.view(n, -1, s)).view(n, p, -1, s)
        qkv = qkv.view(n, p, self.num_head, -1, s).permute(0, 2, 1, 3, 4).contiguous()  # [n x h x p x c x s]
        q, k, v = list(qkv.chunk(3, dim=-2))

        dots = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attention = self.softmax(dots)

        weighted_part_vector = torch.matmul(attention, v.transpose(-2, -1)).transpose(-2,
                                                                                      -1)  # [n x head x p x dim_head x s]
        weighted_part_vector = weighted_part_vector.permute(0, 2, 1, 3, 4).contiguous().view(n, p, -1, s)
        weighted_part_vector = self.to_out(weighted_part_vector.reshape(n, p * c, s)).view(n, p, c, s)
        weighted_part_vector = (weighted_part_vector + t_f).max(-1)[0]

        return weighted_part_vector, attention

    def frame_correlation(self, t_all):
        weighted_part_vector, corrleation = self.frame_corr(t_all)
        weighted_part_vector = (weighted_part_vector + t_all).max(-1)[0]

        return weighted_part_vector, corrleation

    def select_topk_part(self, corr_matrix, t_f):
        n, p, c, s = t_f.shape

        topk_part_index = torch.topk(corr_matrix.sum(-2), self.topk_num, dim=-1, largest=True)[1].unsqueeze(4).repeat(1,
                                                                                                                      1,
                                                                                                                      1,
                                                                                                                      1,
                                                                                                                      c)

        selected_topk_part = torch.zeros_like(t_f[..., 0])

        for i in range(self.num_head):
            selected_topk_part += torch.gather(t_f.transpose(2, 3), dim=2, index=topk_part_index[:, i]).squeeze(2)

        return selected_topk_part

    def forward(self, t_f, t_s=None, t_l=None):
        n, p, c, s = t_f.shape
        t_all = self.decay_c(torch.cat([t_f, t_s, t_l], 2).view(n, -1, s)).view(n, p, c, s)

        weighted_part_vector, attention = self.frame_correlation(t_all)
        selected_topk_part = self.select_topk_part(attention, t_f)
        # plot_heatmap(attention)

        part_feature = []
        for idx, block in enumerate(self.bn):
            part_feature.append(block(weighted_part_vector[:, idx, :]).unsqueeze(0))
        part_feature = torch.cat(part_feature, 0)

        part_classification = part_feature.matmul(self.fc)

        return part_classification.transpose(0, 1), weighted_part_vector.transpose(0, 1), selected_topk_part.transpose(
            0, 1)

