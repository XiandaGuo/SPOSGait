import os
import pickle
import os.path as osp
import torch.utils.data as tordata
import json
from utils import get_msg_mgr


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        ### grew tpami train gxd
        if 'num_train' in data_config.keys():
            num_train = int(data_config['num_train'])
            if num_train != 0 and num_train != 20000:
                msg_mgr.log_info('Note: Only the first {} IDs from the GREW training dataset were used to train the model.\n'.format(num_train) * 5)
                train_set = train_set[:num_train]

        if 'num_test' in data_config:
            num_test = int(data_config['num_test'])
            from gxd.score import get_probe_list
            self.dic_probe = get_probe_list(num_test)
            if num_test != 0:
                msg_mgr.log_info('Note: Only the first {} IDs from the GREW testing dataset were used to train the model.\n'.format(num_test) * 5)
                # msg_mgr.log_info('test_set is {}\n'.format(test_set))
                test_set = test_set[:num_test]
                if 'probe' not in test_set:
                    test_set.append('probe')
                msg_mgr.log_info('tail test_set is {}\n'.format(test_set[-10:]))

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):

                        # 测试1000 2000 4000 8000 16000 ID
                        if lab == 'probe' and 'num_test' in data_config and vie not in self.dic_probe:
                            continue

                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(
            train_set) if training else get_seqs_info_list(test_set)


class DataSet_distr(tordata.Dataset):
    def __init__(self, data_cfg):
        """
            seqs_info: the list with each element indicating
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.__dataset_parser(data_cfg)
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        paths = sorted(paths)
        data_list = []
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                    # 阶段>1000的序列
                    if _.shape[0] > 1000:
                        _ = _[:1000]
                f.close()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config):
        distractor_root = '/mnt/cfs/algorithm/xianda.guo/data/sil/dis-pkl/distractor/04/'
        distr_set = os.listdir(distractor_root)
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if 'num_distr' in data_config:
            num_distr = int(data_config['num_distr'])
            if num_distr != -1:
                msg_mgr.log_info('the number of distr_set is {}\n'.format(num_distr) * 5)
                distr_set = distr_set[:num_distr]


        msg_mgr.log_info("-------- Distractor Pid List --------")
        msg_mgr.log_info(str(len(distr_set)))
        log_pid_list(distr_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            lab, typ = '99999', '04'
            for vie in label_set:
                seq_info = [lab, typ, vie]
                seq_path = osp.join(distractor_root, vie)
                seq_dirs = sorted(os.listdir(seq_path))
                if seq_dirs != []:
                    seq_dirs = [osp.join(seq_path, dir)
                                for dir in seq_dirs]
                    seqs_info_list.append([*seq_info, seq_dirs])
                else:
                    msg_mgr.log_debug(
                        'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        self.seqs_info = get_seqs_info_list(distr_set)
