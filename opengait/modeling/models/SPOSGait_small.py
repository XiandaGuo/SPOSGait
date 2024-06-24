import torch
import torch.nn as nn

from opengait.modeling.models.SPOSGait_large import SPOSGait_large, conv1x1x1, GeMHPP
from opengait.modeling.modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
from opengait.modeling.models.SPOSGait_block import block


class SPOSGait_small(SPOSGait_large):

    def __init__(self, *args, **kargs):
        super(SPOSGait_small, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        in_c = model_cfg['channels']

        # For CASIA-B or other unstated datasets.
        self.relu = nn.ReLU(inplace=True)
        self.conv3d_00 = block(BasicConv3d, 1, in_c[0], flg_relu=True)
        self.downsample0 = nn.Sequential(
            conv1x1x1(1, in_c[0]),
            nn.BatchNorm3d(in_c[0]))
        self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))


        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3d_20 = block(BasicConv3d, in_c[0], in_c[1], flg_relu=True)
        self.downsample2 = nn.Sequential(
            conv1x1x1(in_c[0], in_c[1]),
            nn.BatchNorm3d(in_c[1]))

        self.MaxPool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_30 = block(BasicConv3d, in_c[1], in_c[2], flg_relu=True)
        self.downsample3 = nn.Sequential(
            conv1x1x1(in_c[1], in_c[2]),
            nn.BatchNorm3d(in_c[2]))

        self.conv3d_40 = block(BasicConv3d, in_c[2], in_c[2], flg_relu=True)
        self.downsample4 = nn.Sequential(
            conv1x1x1(in_c[2], in_c[2]),
            nn.BatchNorm3d(in_c[2]))

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP(bin_num=model_cfg['bin_num'])
        self.Head0 = SeparateFCs(**model_cfg['SeparateFCs'])


        loss_cfg = self.cfgs['loss_cfg']
        self.only_use_tripletloss = False
        if len(loss_cfg) < 2 and loss_cfg[0]['type'] == 'TripletLoss':
            self.only_use_tripletloss = True

        if not self.only_use_tripletloss:
            if 'SeparateBNNecks' in model_cfg.keys():  # 正确版本！！！
                self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            else:
                raise ValueError('SeparateBNNecks must be in model_cfg.keys()')


    def forward(self, inputs, cand=None):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)

        # 定义搜索空间
        if cand is None:
            raise RuntimeError('cand can not be none! You must give the cand!')

        identity = sils
        outs = self.conv3d_00[cand[0]](sils)
        residual = self.downsample0(identity)
        outs = outs + residual
        outs = self.relu(outs)
        outs = self.MaxPool0(outs)

        outs = self.LTA(outs)

        identity = outs
        outs = self.conv3d_20[cand[1]](outs)
        residual = self.downsample2(identity)
        outs = outs + residual
        outs = self.MaxPool1(outs)

        identity = outs
        outs = self.conv3d_30[cand[2]](outs)
        residual = self.downsample3(identity)
        outs = outs + residual

        identity = outs
        outs = self.conv3d_40[cand[3]](outs)
        residual = self.downsample4(identity)
        outs = outs + residual  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        embed_1 = self.Head0(outs)  # [p, n, c]
        embed = embed_1

        n, _, s, h, w = sils.size()
        if not self.only_use_tripletloss:
            # BNNechk as Head     # 正确版本！！！
            embed_2, logits = self.BNNecks(embed_1)  # [p, n, c]

            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embed, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs}
                },
                'visual_summary': {
                    'image/sils': sils.view(n * s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }
            return retval

        else:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': embed, 'labels': labs},
                },
                'visual_summary': {
                    'image/sils': sils.view(n * s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }
            return retval


if __name__ == "__main__":
    from opengait.utils.common import config_loader
    import random

    choice_num = [2, 2, 1, 2, 2]
    arch = [random.randint(0, x - 1) for x in choice_num]
    print(arch)

    cfgs = config_loader('/mnt/cfs/algorithm/xianda.guo/code/OpenGait0914/configs/gaitgl_supergait/gaitgl_supergait.yaml')
    torch.distributed.init_process_group('nccl', init_method='env://')
    model = SPOSGait_small(cfgs, training=False)
    test_data = torch.rand(8, 1, 32, 64, 44)
    model(test_data, arch)

