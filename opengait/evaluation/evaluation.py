import os
from time import strftime, localtime
import torch, math
import numpy as np
import torch.nn.functional as F
from utils import get_msg_mgr, mkdir, MeanIOU
from gxd.score import get_scoreBy_localgt
from .metric import mean_iou, cuda_dist, compute_ACC_mAP, evaluate_rank, evaluate_many
from .re_rank import re_ranking

def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
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
        torch.cuda.empty_cache()  # gxd
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

    dist = torch.zeros([num_prob, num_gallery])
    gap_prob = 1000
    gap_gallery = 50000
    for i in range(math.ceil(num_prob / gap_prob)):
        # print(i, math.ceil(num_prob / gap_prob))
        _x = x[i * gap_prob: (i + 1) * gap_prob, :]
        for j in range(math.ceil(num_gallery / gap_gallery)):
            _y = y[j * gap_gallery: (j + 1) * gap_gallery, :]
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(1).transpose(0, 1) \
                    - 2 * torch.matmul(_x, _y.transpose(0, 1))
            _dist = torch.sqrt(F.relu(_dist))
            dist[i * gap_prob: (i + 1) * gap_prob, j * gap_gallery:(j + 1) * gap_gallery] = _dist
            torch.cuda.empty_cache()
    return dist


# Exclude identical-view cases
def de_diag(acc, each_angle=False):
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def identification(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)
    # sample_num = len(feature)

    probe_seq_dict = {'CASIA-B': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']],
                      'OUMVLP': [['00']]}

    gallery_seq_dict = {'CASIA-B': [['nm-01', 'nm-02', 'nm-03', 'nm-04']],
                        'OUMVLP': [['01']]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                    view_num, view_num, num_rank]) - 1.
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        for gallery_seq in gallery_seq_dict[dataset]:
            for (v1, probe_view) in enumerate(view_list):
                for (v2, gallery_view) in enumerate(view_list):
                    gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                        view, [gallery_view])
                    gallery_x = feature[gseq_mask, :]
                    gallery_y = label[gseq_mask]

                    pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                        view, [probe_view])
                    probe_x = feature[pseq_mask, :]
                    probe_y = label[pseq_mask]

                    dist = cuda_dist(probe_x, gallery_x, metric)
                    idx = dist.sort(1)[1].cpu().numpy()
                    acc[p, v1, v2, :] = np.round(
                        np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                               0) * 100 / dist.shape[0], 2)
    result_dict = {}
    np.set_printoptions(precision=3, suppress=True)
    if 'OUMVLP' not in dataset:
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Include identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                np.mean(acc[0, :, :, i]),
                np.mean(acc[1, :, :, i]),
                np.mean(acc[2, :, :, i])))
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: %.3f,\tBG: %.3f,\tCL: %.3f' % (
                de_diag(acc[0, :, :, i]),
                de_diag(acc[1, :, :, i]),
                de_diag(acc[2, :, :, i])))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, i])
        result_dict["scalar/test_accuracy/BG"] = de_diag(acc[1, :, :, i])
        result_dict["scalar/test_accuracy/CL"] = de_diag(acc[2, :, :, i])
        np.set_printoptions(precision=2, floatmode='fixed')
        for i in range(1):
            msg_mgr.log_info(
                '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
            msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, i], True)))
            msg_mgr.log_info('BG: {}'.format(de_diag(acc[1, :, :, i], True)))
            msg_mgr.log_info('CL: {}'.format(de_diag(acc[2, :, :, i], True)))
    else:
        msg_mgr.log_info('===Rank-1 (Include identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (np.mean(acc[0, :, :, 0])))
        msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
        msg_mgr.log_info('NM: %.3f ' % (de_diag(acc[0, :, :, 0])))
        msg_mgr.log_info(
            '===Rank-1 of each angle (Exclude identical-view cases)===')
        msg_mgr.log_info('NM: {}'.format(de_diag(acc[0, :, :, 0], True)))
        result_dict["scalar/test_accuracy/NM"] = de_diag(acc[0, :, :, 0])
    return result_dict


def identification_real_scene(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating by identification_real_scene:")
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0'],
                        'GREW': ['01'], 'TTG-200': ['1']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1'],
                      'GREW': ['02'], 'TTG-200': ['2', '3', '4', '5', '6']}

    num_rank = 20
    acc = np.zeros([num_rank]) - 1.
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()
    acc = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                          0) * 100 / dist.shape[0], 2)
    msg_mgr.log_info('==Rank-1==')
    msg_mgr.log_info('%.5f' % (np.mean(acc[0])))
    msg_mgr.log_info('==Rank-5==')
    msg_mgr.log_info('%.5f' % (np.mean(acc[4])))
    msg_mgr.log_info('==Rank-10==')
    msg_mgr.log_info('%.5f' % (np.mean(acc[9])))
    msg_mgr.log_info('==Rank-20==')
    msg_mgr.log_info('%.5f' % (np.mean(acc[19])))

    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}


def identification_GREW_submission(data, dataset, use_dist=True, metric='euc'):
    get_msg_mgr().log_info("Evaluating GREW")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)
    if use_dist:
        gallery_seq_type = {'GREW': ['01', '02', '04']}
    else:
        gallery_seq_type = {'GREW': ['01', '02']}
    probe_seq_type = {'GREW': ['03']}
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = view[pseq_mask]

    flg_yt = True
    if use_dist and flg_yt:
        pkl_probex = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_probe_x_disall.npy'
        pkl_gallerx = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_gallerx_disall.npy'
        pkl_probey = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_probe_y_disall.npy'
        pkl_gallery = '/mnt/cfs/algorithm/xianda.guo/data/sil/pkl_for_test/pkl_gallery_disall.npy'

        np.save(pkl_probex, probe_x)
        np.save(pkl_gallerx, gallery_x)
        np.save(pkl_probey, probe_y)
        np.save(pkl_gallery, gallery_y)
        raise ValueError('eval distractor by computer_dist.py! ')
    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.cpu().sort(1)[1].numpy()

    flg_roc=False
    if flg_roc:
        dist_numpy = dist.cpu().numpy()
        save_root_roc = '/mnt/nas/algorithm/xianda.guo/data/gait/sposgait-s'
        path_dist_numpy = os.path.join(save_root_roc, 'dist_numpy.npy')
        pkl_gallery = os.path.join(save_root_roc, 'pkl_gallery.npy')
        pkl_probey = os.path.join(save_root_roc, 'pkl_probe_y_disall.npy')
        np.save(path_dist_numpy, dist_numpy)
        np.save(pkl_probey, probe_y)
        np.save(pkl_gallery, gallery_y)
        print('npy for roc have been saved!')

    save_path = os.path.join("GREW_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")

    # if use_dist:
    #     save_path = os.path.join("GREW_result/GaitGLBNNeck/" + 'testID{}_distractor.csv'.format(int(len(idx) / 2)))
    # else:
    #     save_path = os.path.join("GREW_result/GaitGLBNNeck/" + 'testID{}.csv'.format(int(len(idx) / 2)))

    mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write(
            "videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
        for i in range(len(idx)):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:20]]]
            output_row = '{}' + ',{}' * 20 + '\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))

    # evaluation by local gt csv!
    ref_path = '/mnt/nas/algorithm/xianda.guo/data/grew_challenge_gt.csv'
    scores = get_scoreBy_localgt(ref_path, save_path)
    print('all results of rank20 *100 is:', list(scores))

    return


def evaluate_HID(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating HID")
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]

    feat = np.concatenate([probe_x, gallery_x])
    dist = cuda_dist(feat, feat, metric).cpu().numpy()
    msg_mgr.log_info("Starting Re-ranking")
    re_rank = re_ranking(dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
    idx = np.argsort(re_rank, axis=1)

    save_path = os.path.join(
        "HID_result/" + strftime('%Y-%m%d-%H%M%S', localtime()) + ".csv")
    mkdir("HID_result")
    with open(save_path, "w") as f:
        f.write("videoID,label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print("HID result saved to {}/{}".format(os.getcwd(), save_path))
    return


def evaluate_Gait3D(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']
    import json
    probe_sets = json.load(
        open('./datasets/Gait3D/Gait3D.json', 'rb'))['PROBE_SET']
    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
    probe_mask = np.array(probe_mask)

    # probe_features = features[:probe_num]
    probe_features = features[probe_mask]
    # gallery_features = features[probe_num:]
    gallery_features = features[~probe_mask]
    # probe_lbls = np.asarray(labels[:probe_num])
    # gallery_lbls = np.asarray(labels[probe_num:])
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['scalar/test_accuracy/Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100

    # print_csv_format(dataset_name, results)
    msg_mgr.log_info(results)
    return results

def re_ranking(original_dist, query_num, k1, k2, lambda_value):
    # Modified from https://github.com/michuanhaohao/reid-strong-baseline/blob/master/utils/re_ranking.py
    all_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + \
                 original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist


def mean_iou(data, dataset):
    labels = data['mask']
    pred = data['pred']
    miou = MeanIOU(pred, labels)
    get_msg_mgr().log_info('mIOU: %.3f' % (miou.mean()))
    return {"scalar/test_accuracy/mIOU": miou}
