# -*- coding: utf-8 -*-
# @Author  : xianda.guo
# @Time    : 2022/11/23 11:17
# @File    : SPOSGait_block.py
import torch
import torch.nn as nn

def block(BasicConv3d, in_c, out_c, flg_relu=False):
    features = torch.nn.ModuleList()
    if flg_relu:
        for block_index in range(6):
            if block_index == 0:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True))
                )
            elif block_index == 1:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True)
                )
                )
            elif block_index == 2:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True))
                )
            elif block_index == 3:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True)
                )
                )
            elif block_index == 4:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True)
                ))

            elif block_index == 5:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True)
                ))
            else:
                raise NotImplementedError
    else:
        for block_index in range(6):
            if block_index == 0:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.BatchNorm3d(out_c))
                )
            elif block_index == 1:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c)
                )
                )
            elif block_index == 2:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c))
                )
            elif block_index == 3:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c)
                )
                )
            elif block_index == 4:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c)
                ))

            elif block_index == 5:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                    nn.BatchNorm3d(out_c)
                ))
            else:
                raise NotImplementedError

    return features



def block_noBN(BasicConv3d, in_c, out_c, flg_relu=False):
    features = torch.nn.ModuleList()
    if flg_relu:
        for block_index in range(6):
            if block_index == 0:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.ReLU(inplace=True))
                )
            elif block_index == 1:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.ReLU(inplace=True)
                )
                )
            elif block_index == 2:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.ReLU(inplace=True))
                )
            elif block_index == 3:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.ReLU(inplace=True)
                )
                )
            elif block_index == 4:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.ReLU(inplace=True)
                ))

            elif block_index == 5:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                    nn.ReLU(inplace=True)
                ))
            else:
                raise NotImplementedError
    else:
        for block_index in range(6):
            if block_index == 0:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    )
                )
            elif block_index == 1:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                )
                )
            elif block_index == 2:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    )
                )
            elif block_index == 3:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                )
                )
            elif block_index == 4:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                ))

            elif block_index == 5:
                features.append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                ))
            else:
                raise NotImplementedError

    return features



def block_history(features, BasicConv3d, in_c, out_c, flg_relu=False):
    features.append(torch.nn.ModuleList())
    if flg_relu:
        for block_index in range(6):
            if block_index == 0:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True))
                )
            elif block_index == 1:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True)
                )
                )
            elif block_index == 2:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True))
                )
            elif block_index == 3:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True)
                )
                )
            elif block_index == 4:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True)
                ))

            elif block_index == 5:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True)
                ))
            else:
                raise NotImplementedError
    else:
        for block_index in range(6):
            if block_index == 0:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                    nn.BatchNorm3d(out_c))
                )
            elif block_index == 1:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c)
                )
                )
            elif block_index == 2:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c))
                )
            elif block_index == 3:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c)
                )
                )
            elif block_index == 4:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 3), stride=(1, 1, 1), padding=(1, 0, 1)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),
                    nn.BatchNorm3d(out_c)
                ))

            elif block_index == 5:
                features[-1].append(nn.Sequential(
                    BasicConv3d(in_c, out_c,
                           kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
                    nn.BatchNorm3d(out_c),
                    nn.ReLU(inplace=True),
                    BasicConv3d(out_c, out_c,
                           kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                    nn.BatchNorm3d(out_c)
                ))
            else:
                raise NotImplementedError

    return features


