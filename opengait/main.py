import argparse
import os
import sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from modeling import models
from utils import config_loader, get_ddp_module, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--local_rank', type=int, default=0,
                    help="passed by torch.distributed.launch module")
parser.add_argument('--cfgs', type=str,
                    default='config/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', default=True,
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, type=int, help="iter to restore")
parser.add_argument('--num_train', default=0, help="number of ID to train")  # gxd
parser.add_argument('--num_test', default=0, help="number of ID to test")  # gxd
parser.add_argument('--num_distr', default=0, help="number of distrID to test")  # gxd
opt = parser.parse_args()


def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    # gxd
    if 'num_train' in cfgs['data_cfg']:
        output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                                   cfgs['model_cfg']['model'],
                                   engine_cfg['save_name'] + '_' + str(cfgs['data_cfg']['num_train']))
    else:
        output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                                   cfgs['model_cfg']['model'], engine_cfg['save_name'])

    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(cfgs)
    msg_mgr.log_info(cfgs['loss_cfg'])
    msg_mgr.log_info(engine_cfg)

    seed = torch.distributed.get_rank()
    init_seeds(seed)


def run_model(cfgs, local_rank, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, local_rank, training)
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()

    model = get_ddp_module(model, local_rank)

    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    if training:
        Model.run_train(model)
    else:
        if 'SPOSGait' in model_cfg['model']:
            Model.run_test(model, cfgs['retrain']['search_network'])
        else:
            Model.run_test(model)


if __name__ == '__main__':
    torch.distributed.init_process_group('nccl', init_method='env://')
    # torch.distributed.init_process_group('nccl', init_method='tcp://127.0.0.1:2345')
    # if torch.distributed.get_world_size() != torch.cuda.device_count():
    #     raise ValueError("Expect number of availuable GPUs({}) equals to the world size({}).".format(
    #         torch.cuda.device_count(), torch.distributed.get_world_size()))
    cfgs = config_loader(opt.cfgs)

    # 使用多少ID进行训练 for TPami
    if opt.num_train != 0:
        cfgs['data_cfg']['num_train'] = int(opt.num_train)
        cfgs['model_cfg']['SeparateBNNecks']['class_num'] = int(opt.num_train)
        cfgs['model_cfg']['class_num'] = int(opt.num_train)
    # 使用多少ID进行测试 for TPami
    if opt.num_test != 0:
        cfgs['data_cfg']['num_test'] = opt.num_test
    # 使用多少distractor进行测试 for TPami
    if opt.num_distr != 0:
        cfgs['data_cfg']['num_distr'] = opt.num_distr

    if 'SPOSGait' in cfgs['model_cfg']['model'] and cfgs['data_cfg']['dataset_name'] == 'GREW':
        assert cfgs['data_cfg']['num_train'] == cfgs['model_cfg']['SeparateBNNecks']['class_num']
        assert cfgs['model_cfg']['class_num'] == cfgs['model_cfg']['SeparateBNNecks']['class_num']

    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = opt.iter
        cfgs['trainer_cfg']['restore_hint'] = opt.iter

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(opt.local_rank)
    local_rank = opt.local_rank

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, local_rank, training)
