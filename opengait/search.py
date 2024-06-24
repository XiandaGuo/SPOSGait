import argparse
import os
import random
import sys
import time

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

# model
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

# search
parser.add_argument('--log-dir', type=str, default='output')
parser.add_argument('--max-epochs', type=int, default=20)  # 搜索epoch
parser.add_argument('--select-num', type=int, default=10)  #
parser.add_argument('--population-num', type=int, default=50)  # 种群
parser.add_argument('--m_prob', type=float, default=0.1)  # 变异概率
parser.add_argument('--crossover-num', type=int, default=25)  # 交叉
parser.add_argument('--mutation-num', type=int, default=25)  # 变异
parser.add_argument('--flops-limit', type=float, default=330 * 1e6)  # 模型flops限制

opt = parser.parse_args()


class EvolutionSearcher(object):
    def __init__(self, args, local_rank, cfgs):
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_limit = args.flops_limit
        self.cfgs = cfgs
        self.log_dir = args.log_dir
        self.checkpoint_dir = os.path.join(self.log_dir, self.cfgs['data_cfg']['dataset_name'],
                                           self.cfgs['model_cfg']['model'])
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_name = os.path.join(self.checkpoint_dir, os.path.basename(args.cfgs).split('.')[0] + 'search_checkpoint.pth.tar')

        self.Model, self.model = Initialization_model(cfgs, local_rank, training=False)

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], self.population_num: []}
        self.epoch = 0
        self.candidates = []  # 候选

        self.states_of_layers = self.cfgs['model_cfg']['choice_max_num']

    def save_checkpoint(self):
        info = {'memory': self.memory,
                'candidates': self.candidates,
                'vis_dict': self.vis_dict,
                'keep_top_k': self.keep_top_k,
                'epoch': self.epoch}

        rank = torch.distributed.get_rank()
        if rank == 0:
            torch.save(info, self.checkpoint_name)
            print('save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        print('load checkpoint from', self.checkpoint_name)
        return True

    def is_legal(self, cand):
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False
        if 'flops' not in info:
            # info['flops'] = get_cand_flops(cand)
            info['flops'] = np.random.randint(10000)
            print('random flops:', info['flops'])

        if info['flops'] > self.flops_limit:
            print('flops limit exceed')
            return False

        print('is_legal, cand:', cand, 'start to run_test:')
        start_time = time.time()

        res = self.Model.run_test(self.model, cand, flg_search=True)
        print('res', res)
        if self.cfgs['data_cfg']['dataset_name'] =='CASIA-B':
            info['err'] = (res['scalar/test_accuracy/NM'] + res['scalar/test_accuracy/BG'] + res['scalar/test_accuracy/CL'])/3.0
        elif self.cfgs['data_cfg']['dataset_name'] =='GREW':
            info['err'] = res['scalar/test_accuracy/Rank-1']
        else:
            raise ValueError

        end_time = time.time()
        print('inference cost {}s'.format(end_time - start_time))

        info['visited'] = True
        return True

    def get_random(self, num):
        print('random select ........')

        while len(self.candidates) < num:
            cand = [random.randint(0, x - 1) for x in self.states_of_layers]
            cand = tuple(cand)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('-' * 80 + 'epoch-{}'.format(self.epoch) + '--random {}/{}'.format(len(self.candidates), num))
        print('random_num = {}'.format(len(self.candidates)))

    def update_top_k(self, candidates, k, key, reverse=False):
        assert k in self.keep_top_k
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]


    def get_mutation(self, k, mutation_num, m_prob, epoch=0):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        max_iters = mutation_num * 10

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = list(random.choice(self.keep_top_k[k]))
            for i in range(len(self.states_of_layers)):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.states_of_layers[i])
            cand = tuple(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('+' * 80 + 'epoch-{}'.format(epoch) + '--mutation {}/{}'.format(len(res), mutation_num))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num, epoch=0):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            p1 = list(random.choice(self.keep_top_k[k]))
            p2 = list(random.choice(self.keep_top_k[k]))
            cand = [random.choice([i, j]) for i, j in zip(p1, p2)]
            cand = tuple(cand)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('*' * 80 + 'epoch-{}'.format(epoch) + '--crossover {}/{}'.format(len(res), crossover_num))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        # self.load_checkpoint()
        print('start to search:')
        random.seed(0)
        np.random.seed(0)
        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'], reverse=True)
            self.update_top_k(self.candidates, k=self.population_num, key=lambda x: self.vis_dict[x]['err'], reverse=True)

            mutation = self.get_mutation(self.select_num, self.mutation_num, self.m_prob, self.epoch)
            crossover = self.get_crossover(self.select_num, self.crossover_num, self.epoch)

            print('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[self.population_num])))

            for i, cand in enumerate(self.keep_top_k[self.population_num]):
                print('No.{} {} Top-1 err = {}'.format(i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                print(ops)

            # 可改append
            self.candidates = mutation + crossover

            self.get_random(self.population_num)


            self.epoch += 1

        self.save_checkpoint()


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
    if training and cfgs['trainer_cfg']['sync_BN']:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = get_ddp_module(model, local_rank)
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
    if opt.num_test != 0:
        cfgs['data_cfg']['num_test'] = opt.num_test
    if opt.num_distr != 0:
        cfgs['data_cfg']['num_distr'] = opt.num_distr

    gpu_count = torch.cuda.device_count()
    cfgs['evaluator_cfg']['sampler']['batch_size'] = gpu_count

    assert cfgs['evaluator_cfg']['restore_hint'] != 0

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(opt.local_rank)
    local_rank = opt.local_rank

    initialization(cfgs, training=False)
    searcher = EvolutionSearcher(opt, local_rank, cfgs)
    searcher.search()


if __name__ == '__main__':
    main()
