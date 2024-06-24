# -*- coding: utf-8 -*-
# @Author  : xianda.guo
# @Time    : 2022/9/23 14:46
# @File    : comuputer_dist.py
import os
from time import strftime, localtime
import torch, math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    print('cuda_dist:x', x.size())
    print('cuda_dist:y', y.size())
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p

    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
        torch.cuda.empty_cache()
    return 1 - dist / num_bin if metric == 'cos' else dist / num_bin

def cuda_dist_splitc(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    print('cuda_dist:x', x.size())
    print('cuda_dist:y', y.size())
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p

    num_bin = x.size(1)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, i, :]
        _y = y[:, i, :]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
        torch.cuda.empty_cache()
    return 1 - dist / num_bin if metric == 'cos' else dist / num_bin


def cuda_dist_splitp(x, y):
    torch.cuda.empty_cache()
    print('+------+----------+----------+------------+')
    print('only split prob list!')
    print('+------+----------+----------+------------+')
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    num_prob = x.shape[0]
    num_gallery = y.shape[0]

    x = x.reshape(num_prob, -1)
    y = y.reshape(num_gallery, -1)
    print('x', x.size())
    print('y', y.size())

    dist = torch.zeros([num_prob, num_gallery])
    gap_prob = 1000

    for i in range(math.ceil(num_prob / gap_prob)):
        # print(i, math.ceil(num_prob / gap_prob))
        _x = x[i * gap_prob: (i + 1) * gap_prob, :]
        _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(1).transpose(0, 1) \
                - 2 * torch.matmul(_x, y.transpose(0, 1))
        _dist = torch.sqrt(F.relu(_dist))
        dist[i * gap_prob: (i + 1) * gap_prob, :] = _dist
        torch.cuda.empty_cache()
    return dist


def cuda_dist_split_pg(x, y):
    print('+------+----------+----------+------------+')
    print('split both prob and gallery list!')
    print('+------+----------+----------+------------+')
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    num_prob = x.shape[0]
    num_gallery = y.shape[0]

    x = x.reshape(num_prob,-1)
    y = y.reshape(num_gallery,-1)
    print('cuda_dist_split_pg:x', x.size())
    print('cuda_dist_split_pg:y', y.size())


    dist = torch.zeros([num_prob, num_gallery])
    gap_prob = 1000
    gap_gallery = 50000
    for i in tqdm(range(math.ceil(num_prob / gap_prob))):
        # print(i, math.ceil(num_prob / gap_prob))
        _x = x[i * gap_prob: (i + 1) * gap_prob, :]
        for j in range(math.ceil(num_gallery / gap_gallery)):
            _y = y[j * gap_gallery: (j + 1) * gap_gallery, :]
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
                0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            _dist = torch.sqrt(F.relu(_dist))
            dist[i * gap_prob: (i + 1) * gap_prob, j * gap_gallery:(j + 1) * gap_gallery] = _dist
            torch.cuda.empty_cache()
    print('the function of computing dist is finished!')
    print('dist.size', dist.size())
    return dist


def main(num_dis):
    # pkl_probex = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_probe_x.npy'
    # pkl_gallerx = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_gallerx.npy'
    # pkl_probey = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_probe_y.npy'
    # pkl_gallery = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_gallery.npy'

    pkl_probex = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_probe_x_disall.npy'
    pkl_gallerx = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_gallerx_disall.npy'
    pkl_probey = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_probe_y_disall.npy'
    pkl_gallery = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_gallery_disall.npy'

    print('start to load npy:')
    probe_x = np.load(pkl_probex)
    probe_y = np.load(pkl_probey)
    gallery_x = np.load(pkl_gallerx)
    gallery_y = np.load(pkl_gallery)
    print('finised!')

    gallery_x =gallery_x[:12000+num_dis]
    gallery_y =gallery_y[:12000+num_dis]
    print('gallery_x', gallery_x.shape)
    print('gallery_y', gallery_y.shape)
    # dist = cuda_dist(probe_x, gallery_x)
    # dist = cuda_dist_splitc(probe_x, gallery_x)
    # dist = cuda_dist_splitp(probe_x, gallery_x)
    # idx = dist.cpu().sort(1)[1].numpy()

    dist = cuda_dist_split_pg(probe_x, gallery_x)
    idx = dist.sort(1)[1].cpu().numpy()

    save_path = os.path.join(
        "/mnt/cfs/algorithm/xianda.guo/code/OpenGait0914/GREW_result/test_dis/" + strftime('%Y-%m%d-%H%M%S', localtime()) + ".csv")

    # mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write(
            "videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
        for i in tqdm(range(len(idx))):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:20]]]
            output_row = '{}' + ',{}' * 20 + '\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))

    # evaluation in local gt csv!
    from opengait.gxd.score import get_scoreBy_localgt
    ref_path = '/mnt/nas/algorithm/xianda.guo/data/grew_challenge_gt.csv'
    scores = get_scoreBy_localgt(ref_path, save_path)
    print(num_dis, '---all results of rank20:', list(scores))
    torch.cuda.empty_cache()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    use_num_distractors =[50000, 100000, 150000, 200000, 240000] #
    # use_num_distractor =240000 #
    for use_num_distractor in use_num_distractors:
        if use_num_distractor!=50000:
            break #
        main(use_num_distractor)
