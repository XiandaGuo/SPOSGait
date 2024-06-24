import torch
import numpy as np
import sys
import numpy as np
import random
import torchvision.transforms as T
import cv2
import math
import torchvision.transforms.functional as F
from PIL import Image
import torchvision.transforms as T

from data import transform as base_transform
from utils import is_list, is_dict, get_valid_args


class NoOperation():
    def __call__(self, x):
        return x


class BaseSilTransform():
    def __init__(self, disvor=255.0, img_shape=None):
        self.disvor = disvor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.disvor


class BaseSilCuttingTransform():
    def __init__(self, img_w=64, disvor=255.0, cutting=None):
        self.img_w = img_w
        self.disvor = disvor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(self.img_w // 64) * 10
        x = x[..., cutting:-cutting]
        return x / self.disvor


class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        if std is None:
            std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        self.mean = np.array(mean).reshape((1, 3, 1, 1))
        self.std = np.array(std).reshape((1, 3, 1, 1))

    def __call__(self, x):
        return (x - self.mean) / self.std


class RandomRotateFlipTransform():
    def __init__(self, img_w):
        self.BaseSilCuttingTransform = BaseSilCuttingTransform(img_w)
        self.rotate_p = 0.5
        self.flip_p = 0.5

    def __call__(self, seqs):
        seqs = self.BaseSilCuttingTransform(seqs)
        aug_seqs = []

        degrees = [-10, 10]
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        flg_rotate = True if torch.rand(1) < self.rotate_p else False
        flg_flip = True if torch.rand(1) < self.rotate_p else False
        for fram in seqs:

            fram = Image.fromarray(fram)

            if flg_flip:
                fram = F.hflip(fram)
            if flg_rotate:
                fram = F.rotate(fram, angle)

            fram = np.array(fram)
            aug_seqs.append(fram)

        return np.array(aug_seqs)


class RandomFlip_gxd():
    def __init__(self, prob=0.5):
        self.flip_p = prob

    def __call__(self, seq):
        flg_flip = True if torch.rand(1) < self.flip_p else False
        if flg_flip:
            aug_seqs = []
            for fram in seq:
                fram = Image.fromarray(fram)
                if flg_flip:
                    fram = F.hflip(fram)
                fram = np.array(fram)
                aug_seqs.append(fram)

            return np.array(aug_seqs)
        else:
            return seq

class RandomRotate_gxd(object):
    def __init__(self, prob=0.5, degree=10):
        self.rotate_p = prob
        self.degree = degree
    def __call__(self, seq):
        flg_rotate = True if torch.rand(1) < self.rotate_p else False
        if flg_rotate:
            aug_seqs = []
            degrees = [-self.degree, self.degree]
            angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
            for fram in seq:
                fram = Image.fromarray(fram)
                if flg_rotate:
                    fram = F.rotate(fram, angle)
                fram = np.array(fram)
                aug_seqs.append(fram)
            return np.array(aug_seqs)
        else:
            return seq

# **************** Data Agumentation OpenGait****************
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            return seq[..., ::-1]


class RandomErasing(object):
    def __init__(self, prob=0.5, sl=0.05, sh=0.2, r1=0.3, per_frame=False):
        self.prob = prob
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.per_frame = per_frame

    def __call__(self, seq):
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                for _ in range(100):
                    seq_size = seq.shape
                    area = seq_size[1] * seq_size[2]

                    target_area = random.uniform(self.sl, self.sh) * area
                    aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))

                    if w < seq_size[2] and h < seq_size[1]:
                        x1 = random.randint(0, seq_size[1] - h)
                        y1 = random.randint(0, seq_size[2] - w)
                        seq[:, x1:x1 + h, y1:y1 + w] = 0.
                        return seq
            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...])
                   for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)


class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                  for _ in seq], 0)
            return seq


class RandomPerspective(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, h, w = seq.shape
            cutting = int(w // 44) * 10
            x_left = list(range(0, cutting))
            x_right = list(range(w - cutting, w))
            TL = (random.choice(x_left), 0)
            TR = (random.choice(x_right), 0)
            BL = (random.choice(x_left), h)
            BR = (random.choice(x_right), h)
            srcPoints = np.float32([TL, TR, BR, BL])
            canvasPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            perspectiveMatrix = cv2.getPerspectiveTransform(
                np.array(srcPoints), np.array(canvasPoints))
            seq = [cv2.warpPerspective(_[0, ...], perspectiveMatrix, (w, h))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                  for _ in seq], 0)
            return seq


class RandomAffine(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            _, dh, dw = seq.shape
            # rotation
            max_shift = int(dh // 64 * 10)
            shift_range = list(range(0, max_shift))
            pts1 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                dh - random.choice(shift_range), random.choice(shift_range)],
                               [random.choice(shift_range), dw - random.choice(shift_range)]])
            pts2 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                dh - random.choice(shift_range), random.choice(shift_range)],
                               [random.choice(shift_range), dw - random.choice(shift_range)]])
            M1 = cv2.getAffineTransform(pts1, pts2)
            # affine
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                  for _ in seq], 0)
            return seq


# ******************************************

def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform

def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"


if __name__ == '__main__':

    import pickle, cv2
    # img = torch.randn((30,64,64))
    pth = '/mnt/nas/public_data/GREW/GREW-pkl/00001train/00/4XPn5Z28/4XPn5Z28.pkl'
    with open(pth, 'rb') as f:
        seqs = pickle.load(f)
    f.close()

    print(type(seqs))
    print(seqs.shape)
    base_args = {'img_w': 64}

    for epoch in range(5):
        if epoch>0:
            break
        print('epoch', epoch)
        # transform = RandomRotateFlipTransform(64)
        trf_cfg = [{'type':'BaseSilCuttingTransform'}, {'type':'RandomRotate_gxd', 'prob':0.5}]
        transform = Compose(trf_cfg)
        aug_seqs = transform(seqs)*255.0
        for i in range(seqs.shape[0]):
            # if i > 0:
            #     break
            print(i, np.unique(seqs[i]))
            print(i, np.unique(aug_seqs[i]))

            cv2.imwrite('/mnt/nas/algorithm/xianda.guo/data/debug-dataaug/img-e{}-{}.jpg'.format(epoch, i), seqs[i])
            cv2.imwrite('/mnt/nas/algorithm/xianda.guo/data/debug-dataaug/aug_img-e{}-{}.jpg'.format(epoch, i), aug_seqs[i])

        print('finish!')
    from torchvision import transforms
