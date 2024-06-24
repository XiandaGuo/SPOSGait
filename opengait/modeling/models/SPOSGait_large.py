import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from opengait.utils import Odict, ddp_all_gather
from opengait.utils import get_valid_args, ts2np
from opengait.evaluation import evaluation as eval_functions

from opengait.modeling.base_model import BaseModel
from opengait.modeling.modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
from opengait.modeling.models.SPOSGait_block import block


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class GeMHPP(nn.Module):
    def __init__(self, bin_num=[64], p=6.5, eps=1.0e-6):
        super(GeMHPP, self).__init__()
        self.bin_num = bin_num
        self.p = nn.Parameter(
            torch.ones(1) * p)
        self.eps = eps

    def gem(self, ipts):
        return F.avg_pool2d(ipts.clamp(min=self.eps).pow(self.p), (1, ipts.size(-1))).pow(1. / self.p)

    def forward(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = self.gem(z).squeeze(-1)
            features.append(z)
        return torch.cat(features, -1)


class SPOSGait_large(BaseModel):

    def __init__(self, *args, **kargs):
        super(SPOSGait_large, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        self.model_cfg = model_cfg
        in_c = model_cfg['channels']
        dataset_name = self.cfgs['data_cfg']['dataset_name']

        assert dataset_name in ['OUMVLP', 'GREW', 'Gait3D']

        # For OUMVLP, GREW and Gait3D
        self.relu = nn.ReLU(inplace=True)
        self.conv3d_00 = block(BasicConv3d, 1, in_c[0], flg_relu=True)
        self.conv3d_01 = block(BasicConv3d, in_c[0], in_c[0])
        self.downsample0 = nn.Sequential(
            conv1x1x1(1, in_c[0]),
            nn.BatchNorm3d(in_c[0]))
        self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_10 = block(BasicConv3d, in_c[0], in_c[1], flg_relu=True)
        self.conv3d_11 = block(BasicConv3d, in_c[1], in_c[1])
        self.downsample1 = nn.Sequential(
            conv1x1x1(in_c[0], in_c[1]),
            nn.BatchNorm3d(in_c[1]))

        self.LTA = nn.Sequential(
            BasicConv3d(in_c[1], in_c[1], kernel_size=(
                3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3d_20 = block(BasicConv3d, in_c[1], in_c[2], flg_relu=True)
        self.conv3d_21 = block(BasicConv3d, in_c[2], in_c[2])
        self.downsample2 = nn.Sequential(
            conv1x1x1(in_c[1], in_c[2]),
            nn.BatchNorm3d(in_c[2]))

        self.MaxPool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_30 = block(BasicConv3d, in_c[2], in_c[3], flg_relu=True)
        self.conv3d_31 = block(BasicConv3d, in_c[3], in_c[3])
        self.downsample3 = nn.Sequential(
            conv1x1x1(in_c[2], in_c[3]),
            nn.BatchNorm3d(in_c[3]))

        self.conv3d_40 = block(BasicConv3d, in_c[3], in_c[3], flg_relu=True)
        self.conv3d_41 = block(BasicConv3d, in_c[3], in_c[3])
        self.downsample4 = nn.Sequential(
            conv1x1x1(in_c[3], in_c[3]),
            nn.BatchNorm3d(in_c[3]))

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
        # print('only_use_tripletloss', self.only_use_tripletloss)


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

        assert self.cfgs['data_cfg']['dataset_name'] in ['OUMVLP', 'GREW', 'Gait3D']
        identity = sils
        outs = self.conv3d_00[cand[0]](sils)
        outs = self.conv3d_01[cand[1]](outs)
        residual = self.downsample0(identity)
        outs += residual
        outs = self.relu(outs)
        outs = self.MaxPool0(outs)

        identity = outs
        outs = self.conv3d_10[cand[2]](outs)
        outs = self.conv3d_11[cand[3]](outs)
        residual = self.downsample1(identity)
        outs += residual
        outs = self.relu(outs)

        outs = self.LTA(outs)

        identity = outs
        outs = self.conv3d_20[cand[4]](outs)
        outs = self.conv3d_21[cand[5]](outs)
        residual = self.downsample2(identity)
        outs += residual
        outs = self.MaxPool1(outs)

        identity = outs
        outs = self.conv3d_30[cand[6]](outs)
        outs = self.conv3d_31[cand[7]](outs)
        residual = self.downsample3(identity)
        outs += residual

        identity = outs
        outs = self.conv3d_40[cand[8]](outs)
        outs = self.conv3d_41[cand[9]](outs)
        residual = self.downsample4(identity)
        outs += residual  # [n, c, s, h, w]


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

    @staticmethod
    def run_train(model):
        """Accept the instance object(model) here, and then run the train loop."""
        for i, inputs in enumerate(model.train_loader):
            # print('inputs:', inputs)
            random.seed(i)
            ipts = model.inputs_pretreament(inputs)
            with autocast(enabled=model.engine_cfg['enable_float16']):
                # debug
                if model.model_cfg['choice_debug']:
                    rank = torch.distributed.get_rank()
                    print('i', i, 'rank', rank, 'architecture', architecture)

                if 'retrain' in model.cfgs.keys():
                    if i == 5:
                        print('supergait_retrain in model.cfgs, start to retrain the network!')
                    architecture = model.cfgs["retrain"]['search_network']
                else:
                    # raise ValueError('Attention!!! only used in nas train!!!')
                    choice_max_num = model.model_cfg['choice_max_num']
                    architecture = [random.randint(0, x - 1) for x in choice_max_num]

                retval = model(ipts, architecture)
                training_feat, visual_summary = retval['training_feat'], retval['visual_summary']
                del retval
            loss_sum, loss_info = model.loss_aggregator(training_feat)
            ok = model.train_step(loss_sum)
            if not ok:
                continue

            visual_summary.update(loss_info)
            visual_summary['scalar/learning_rate'] = model.optimizer.param_groups[0]['lr']

            _lr = model.optimizer.param_groups[0]['lr']
            loss_info['scalar/lr'] = _lr
            model.msg_mgr.train_step(loss_info, visual_summary)


            if model.iteration % model.engine_cfg['save_iter'] == 0:
                # save the checkpoint
                model.save_ckpt(model.iteration)

                # run test if with_test = true
                if model.engine_cfg['with_test']and model.iteration>50000:
                    model.msg_mgr.log_info("Running test...")
                    model.eval()
                    result_dict = model.run_test(model, architecture)
                    model.train()
                    if model.cfgs['trainer_cfg']['fix_BN']:
                        model.fix_BN()
                    # model.msg_mgr.write_to_tensorboard(result_dict)
                    model.msg_mgr.reset_time()
            if model.iteration >= model.engine_cfg['total_iter']:
                break

    # for search
    def inference(self, rank, cand):
        """Inference all the test data.

        Args:
            rank: the rank of the current process.Transform
        Returns:
            Odict: contains the inference results.
        """
        total_size = len(self.test_loader)
        batch_size = self.test_loader.batch_sampler.batch_size
        rest_size = total_size
        info_dict = Odict()
        for inputs in self.test_loader:
            ipts = self.inputs_pretreament(inputs)
            with autocast(enabled=self.engine_cfg['enable_float16']):
                retval = self.forward(ipts, cand)
                inference_feat = retval['inference_feat']
                for k, v in inference_feat.items():
                    inference_feat[k] = ddp_all_gather(v, requires_grad=False)
                del retval
            for k, v in inference_feat.items():
                inference_feat[k] = ts2np(v)
            info_dict.append(inference_feat)
            rest_size -= batch_size

            torch.cuda.empty_cache()

        # pbar.close()
        for k, v in info_dict.items():
            v = np.concatenate(v)[:total_size]
            info_dict[k] = v

        label_list, types_list, views_list = self.list_all
        info_dict.update({
            'labels': label_list, 'types': types_list, 'views': views_list})

        return info_dict

    # for search
    @staticmethod
    def run_test(model, cand, flg_search=False):
        """Accept the instance object(model) here, and then run the test loop."""

        rank = torch.distributed.get_rank()
        with torch.no_grad():
            info_dict = model.inference(rank, cand)

        if rank == 0:
            if 'eval_func' in model.cfgs["evaluator_cfg"].keys():
                eval_func = model.cfgs['evaluator_cfg']["eval_func"]
            else:
                eval_func = 'identification'
            eval_func = getattr(eval_functions, eval_func)
            valid_args = get_valid_args(
                eval_func, model.cfgs["evaluator_cfg"], ['metric'])
            try:
                dataset_name = model.cfgs['data_cfg']['test_dataset_name']
            except:
                dataset_name = model.cfgs['data_cfg']['dataset_name']

            if model.cfgs["evaluator_cfg"]["eval_func"] == 'identification_GREW_submission':
                return eval_func(info_dict, dataset_name, model.use_dis, **valid_args)
            else:
                return eval_func(info_dict, dataset_name, **valid_args)
        else:
            if flg_search:
                if 'eval_func' in model.cfgs["evaluator_cfg"].keys():
                    eval_func = model.cfgs['evaluator_cfg']["eval_func"]
                else:
                    eval_func = 'identification'
                eval_func = getattr(eval_functions, eval_func)
                valid_args = get_valid_args(
                    eval_func, model.cfgs["evaluator_cfg"], ['metric'])
                try:
                    dataset_name = model.cfgs['data_cfg']['test_dataset_name']
                except:
                    dataset_name = model.cfgs['data_cfg']['dataset_name']

                if model.cfgs["evaluator_cfg"]["eval_func"] == 'identification_GREW_submission':
                    return eval_func(info_dict, dataset_name, model.use_dis, **valid_args)
                else:
                    return eval_func(info_dict, dataset_name, **valid_args)