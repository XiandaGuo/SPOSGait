import sys
import argparse
import os
import random
import sys
import time
import thop
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.getcwd())

# from helper import get_cand_err
from modeling import models
# from modeling.models.gaitgl_supergait import SuperGait
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser(description='Main program for opengait.')

# opengait
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
parser.add_argument('--num_train', default=0, help="number of ID to train") #gxd
parser.add_argument('--num_test', default=0, help="number of ID to test") #gxd
parser.add_argument('--num_distr', default=0, help="number of distrID to test") #gxd

opt = parser.parse_args()



def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])

    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(cfgs)
    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def Initialization_model(cfgs, local_rank, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, local_rank, training)

    model.eval()

    flg_casib=False
    if flg_casib:
        ipts = [1, 1, 1186, 64, 44]
        # labs = [16]
        labs = [1] #gaitgl
        seqL = [1, 16]
        tmp = [16]
    else:
        ipts = [1, 1, 1186, 64, 44]
        # labs = [16]
        labs = [1]  # gaitgl
        seqL = [1, 16]
        tmp = [16]
    inputs = [torch.randn(ipts).cuda(), torch.randint(high=2, size=labs).cuda(),
              torch.randn(tmp).cuda(), torch.randn(tmp).cuda(), torch.randint(low=1, high=74, size=seqL).cuda()]

    print('model', model)
    if 'SPOS' in cfgs['model_cfg']['model']:
        # cand = [0, 0, 0, 1] #casi-b
        cand = cfgs['retrain']['search_network']   #grew
        flops, params = thop.profile(model, inputs=(inputs, cand))
    else:
        flops, params = thop.profile(model, inputs=(inputs,))
    print("Number of calculates:%.2fGFlops" % (flops / 1e9))
    print('Params: %.2f M' % (params / 1e6))

    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    return Model, model


def main():
    torch.distributed.init_process_group('nccl', init_method='env://')
    # torch.distributed.init_process_group('nccl', init_method='tcp://127.0.0.1:2345')
    if torch.distributed.get_world_size() != torch.cuda.device_count():
        raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
            torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)
    if opt.num_train != 0:
        cfgs['data_cfg']['num_train'] = opt.num_train
        # cfgs['model_cfg']['class_num'] = opt.num_train
    if opt.num_test != 0:
        cfgs['data_cfg']['num_test'] = opt.num_test
    if opt.num_distr != 0:
        cfgs['data_cfg']['num_distr'] = opt.num_distr

    print('loss_cfg', cfgs['loss_cfg'])
    # cfgs['loss_cfg'] = cfgs['loss_cfg'][:1]
    print('loss_cfg', cfgs['loss_cfg'])
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(opt.local_rank)
    local_rank = opt.local_rank

    initialization(cfgs, training=False)
    Initialization_model(cfgs, local_rank, training=False)


if __name__ == '__main__':
    main()