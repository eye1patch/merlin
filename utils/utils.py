import os
import shutil
import glob
import numpy as np
import torch.nn.functional as F
from datasets import Features, Value, Sequence
from einops import rearrange

import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader

import torch
import torch.distributed as dist
import logging

import copy

from safetensors.torch import load_file

IGNORE_INDEX = 3

logger = logging.getLogger(__name__)


def pack_sequence(samples, patch_size, max_seq_len, enable_packing=True):
    """
    打包函数，传入长短序列的可迭代对象，传出打包好的tensor，以及两张辅助表
    输入
    samples: 二维列表，元素是三维tensor
    patch_size: tuple 表示patch形状，未来要改成一维形式表示序列最大长度
    max_seq_len: int 整条序列长度
    输出
    patches: tensor (bs, seq_len, 2*ph*pw) seq_len对齐
    patch_positions: tensor (bs, seq_len, 2) 2指ij坐标 用于位置编码计算
    attn_mask: tensor (bs, seq_len, seq_len) 用于attn层计算
    """
    max_patches_len = max_seq_len // patch_size  # 16
    sequences = samples  # (2, 144, 1)

    sample_ids_seq = []  # 用于计算attn_mask的样本边界
    patches_seq = []  # 将转化为打包序列主体
    positions_seq = []  # 用于位置编码的坐标
    for iq_seq in sequences:
        current_patches = []
        positions = []
        sample_ids = torch.empty((0,), dtype=torch.long)
        for sample_id, sample in enumerate(iq_seq):
            sample_len = sample.shape[-2]  # 144
            num_patches = sample_len // patch_size  # 计算patch数
            pos = torch.arange(num_patches)  # 生成位置坐标
            # 数据分块处理（通道维度合并） (2, seq_len, 1) -> (seq_len/patch_size, 2*patch_size) (18, 16)
            iq_patch = rearrange(
                sample, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=patch_size, p2=1
            )
            # 记录样本归属ID
            sample_ids = F.pad(sample_ids, (0, iq_patch.shape[-2]), value=sample_id)
            current_patches.append(iq_patch)
            positions.append(pos)
        # 合并当前序列组的结果
        sample_ids_seq.append(sample_ids)
        patches_seq.append(torch.cat(current_patches, dim=0))
        positions_seq.append(torch.cat(positions, dim=0))

    # 用零补齐所有序列到最大长度 max_patches_len
    # patches(bs, max_patches_num, patches_dim) patch_pos(bs, max_patches_num)
    patches = torch.stack(  # (618, 16, 16)
        [
            F.pad(patch, (0, 0, 0, max_patches_len - len(patch)), value=0)
            for patch in patches_seq
        ],
        dim=0,
    )

    patch_positions = torch.stack(
        [F.pad(pos, (0, max_patches_len - len(pos)), value=0) for pos in positions_seq],
        dim=0,
    )

    # # gwj 0522: 尝试将sample_ids_seq从列表转为tensor
    # sample_ids_seq = torch.stack(sample_ids_seq, dim=0)  # (bs, max_patches_num)

    return patches, patch_positions, sample_ids_seq

def transform_downstream(
    batch,
    patch_size,
    iq_column_name,
    norm_method_iq="abs"  # Gabrielle 0528：默认设为None表示不处理回归参数
):
    # IQ数据归一化
    # iq_data = np.array(batch[sit_column_name], dtype=np.float32)  # (bs, n, 2)

    # 同一条数据中不同IQ长度可能不同，分开处理
    iq_data = []
    for sample in batch[iq_column_name]:
        sample_list = []
        for item in sample:
            # 不同 item 可能是 list of varying length
            norm_sample = normalize_data(item, norm_method=norm_method_iq)
            sample_list.append(norm_sample)
        iq_data.append(sample_list)
    # iq_data = normalize_data(batch[iq_column_name], norm_method=norm_method_iq) # (bs, n, 2)

    iq_data = np.transpose(np.clip(iq_data, -5.0, 5.0), (0, 1, 3, 2))[
        ..., np.newaxis
    ]  # (bs, n_sample, 2, n, 1)

    # 需要在signal前加cls token，以便encoder发挥最好性能，在得到encoder结果后，cls token将被丢弃
    cls_token = np.zeros((iq_data.shape[0], iq_data.shape[1], 2, patch_size[0], 1))  # (bs, n_samples, 2, 8, 1)

    iq_data = np.concatenate((cls_token, iq_data), axis=3)  # (bs, n_samples, 2, n+8, 1)

    batch[iq_column_name] = iq_data

    return batch


# def transform_downstream_optimized(
#     batch,
#     patch_size,
#     iq_column_name,
#     norm_method_iq="abs"
# ):
#     # 把 batch[iq_column_name] 转成 ndarray，保证 shape = (bs, n_samples, n_points, 2)
#     # 注意：如果每个 sample 的长度不同，可以先 pad 到同样长度
#     iq_list = batch[iq_column_name]

#     # 找到最长序列，用 0 pad
#     max_len = max(max(len(item) for item in sample) for sample in iq_list)
#     bs = len(iq_list)
#     n_samples = [len(sample) for sample in iq_list]

#     # 用 nan 填充，方便后面统一处理
#     padded = np.full((bs, max(n_samples), max_len, 2), np.nan, dtype=np.float32)
#     for i, sample in enumerate(iq_list):
#         for j, item in enumerate(sample):
#             arr = np.asarray(item, dtype=np.float32)
#             padded[i, j, :arr.shape[0], :] = arr

#     # 归一化
#     normed = normalize_data(padded, norm_method=norm_method_iq)  # (bs, n_sample, n, 2)

#     # clip + transpose + expand
#     normed = np.clip(normed, -5.0, 5.0)
#     normed = np.transpose(normed, (0, 1, 3, 2))[..., np.newaxis]  
#     # (bs, n_sample, 2, n, 1)

#     # 加 cls token
#     cls_token = np.zeros((normed.shape[0], normed.shape[1], 2, patch_size[0], 1), dtype=np.float32)
#     normed = np.concatenate((cls_token, normed), axis=3)  

#     batch[iq_column_name] = normed
#     return batch


def normalize_data(data, norm_method="std"):
    """
    默认为输入批次数据，如果没有批次则应该先处理为批次后送入
    data可能包括两种情况：iq or 回归参数
    iq.shape (bs, n, 2)
    回归参数.shape (bs,)
    回归参数为一维数据
    """

    # 将输入数据转换为 NumPy 数组
    if isinstance(data, list):
        data = np.array(data, dtype=np.float32)
    elif isinstance(data, torch.Tensor):
        data = data.numpy()
    elif not isinstance(data, np.ndarray):
        raise ValueError("输入数据必须是列表、NumPy 数组或 PyTorch Tensor")

    # # print(f"输入数据的类型: {type(data)}")
    # if np.isnan(data).any() or np.isinf(data).any():
    #     import pdb
    #     pdb.set_trace()
    #     print(f"Null in normalize_data")  # 预计检测回归参数中的null
    #     return None

    # SR任务中，可能存在大片值为None的采样点
    data = np.where(data == None, np.nan, data).astype(np.float32)

    # mask = (~np.isnan(data)).astype(np.float32)

    # if np.isnan(data).all() or 0 < np.isnan(data).mean() < 0.1:
    #     raise ValueError("Nan data")


    assert data.ndim == 2
    norm_axis = tuple(range(0, data.ndim))
    # # 动态生成归一化的轴
    # if data.ndim == 1:
    #     norm_axis = 0
    # else:
    #     norm_axis = tuple(range(2, data.ndim))  # iq应该两列一起算

    if norm_method == "std":
        mean = np.nanmean(data, axis=norm_axis, keepdims=True)
        std = np.nanstd(data, axis=norm_axis, keepdims=True)
        std = np.clip(std, a_min=1e-6, a_max=None)
        normalized_data = (data - mean) / std
    elif norm_method == "abs":  # 绝对幅值归一化
        norm = np.nanmax(np.abs(data), axis=norm_axis, keepdims=True)
        normalized_data = data / (norm + 1e-6)
    else:
        raise ValueError("不支持的归一化方法，仅支持 'std' 或 'abs'")

    # 将结果转换回原始类型
    if isinstance(data, list):
        return normalized_data.tolist()
    elif isinstance(data, torch.Tensor):
        return torch.tensor(normalized_data, dtype=data.dtype)
    else:
        return normalized_data


def transform_downstream(
    batch,
    patch_size,
    iq_column_name,
    norm_method_iq="abs"  # Gabrielle 0528：默认设为None表示不处理回归参数
):
    # IQ数据归一化
    # iq_data = np.array(batch[sit_column_name], dtype=np.float32)  # (bs, n, 2)

    # 同一条数据中不同IQ长度可能不同，分开处理
    iq_data = []
    for sample in batch[iq_column_name]:
        sample_list = []
        for item in sample:
            # 不同 item 可能是 list of varying length
            norm_sample = normalize_data(item, norm_method=norm_method_iq)
            sample_list.append(norm_sample)
        iq_data.append(sample_list)
    # iq_data = normalize_data(batch[iq_column_name], norm_method=norm_method_iq) # (bs, n, 2)
    try:
        iq_data = np.transpose(np.clip(iq_data, -5.0, 5.0), (0, 1, 3, 2))[
            ..., np.newaxis
        ]  # (bs, n_sample, 2, n, 1)
    except:
        print(len(iq_data), len(iq_data[0]), len(iq_data[0][0]), len(iq_data[0][0][0]))
        import pdb; pdb.set_trace()

    # 需要在signal前加cls token，以便encoder发挥最好性能，在得到encoder结果后，cls token将被丢弃
    cls_token = np.zeros((iq_data.shape[0], iq_data.shape[1], 2, patch_size[0], 1))  # (bs, n_samples, 2, 8, 1)

    iq_data = np.concatenate((cls_token, iq_data), axis=3)  # (bs, n_samples, 2, n+8, 1)

    batch[iq_column_name] = iq_data

    return batch