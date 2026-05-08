# zzy 0520 新增

# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
# Which was adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

"""Helper functions for padding and unpadding batches.

These functions are used extensively throughout the Mosaic BERT implementation
in `bert_layers.py`.
"""

from typing import Tuple, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# 定义一个自定义的autograd函数，用于在第一维上根据索引取出元素
class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """提取input中第一维上在indices指定位置的元素。

        参数:
            ctx: autograd上下文对象，用于保存反向传播所需的信息
            input: (b, ...) 具有至少2维的张量
            indices: (num_idx) 一维张量，表示要选取的索引
        """
        ctx.save_for_backward(indices)  # 保存索引以备反向传播使用
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]  # 保存原始input的第一维大小和剩余形状
        second_dim = other_shape.numel()  # 剩余各维度元素个数的乘积

        # 使用einops将input展平成 (b, second_dim)，在第一维上gather索引
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),  # (b, ...) -> (b, second_dim)
            0,
            repeat(indices, "z -> z d", d=second_dim),  # 将indices扩展为 (indices, second_dim)
        ).reshape(-1, *other_shape)  # 恢复为 (num_idx, ...)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """反向传播，构建梯度"""
        (indices,) = ctx.saved_tensors  # 取出保存的indices
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")  # 展平成2维

        # 初始化与input同形状的梯度张量
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype
        )

        # 将grad_output根据indices散布到grad_input对应位置
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)

        # 恢复原来的形状
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None

# 将IndexFirstAxis包装成便捷函数
index_first_axis = IndexFirstAxis.apply

# 定义另一个自定义autograd函数，用于在第一维上根据索引放置元素
class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:
        """将values根据indices放置到新张量中。

        参数:
            values: (num_idx, ...) 输入张量
            indices: (num_idx) 索引
            first_axis_dim: 新张量第一维的大小
        """
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2

        # 初始化一个全零张量
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)

        # 根据indices将values写入output
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """反向传播，只对values求梯度"""
        (indices,) = ctx.saved_tensors
        # 取出grad_output中对应indices位置的部分
        grad_values = grad_output[indices]
        return grad_values, None, None

# 将IndexPutFirstAxis包装成便捷函数
index_put_first_axis = IndexPutFirstAxis.apply

# modernBERT中的，参考
def unpad_input_refer(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """移除输入序列中的padding。

    参数:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen)，布尔或整型张量，1表示有效，0表示无效

    返回:
        hidden_states: (total_nnz, ...) ，去除padding后的张量
        indices: (total_nnz)，有效token的位置
        cu_seqlens: (batch + 1)，每个序列的累计长度（用于快速定位）
        max_seqlen_in_batch: 批次中的最大有效序列长度
    """
    # 每个序列中有效token的数量
    # [[1,1,1,1,0],[1,1,1,1,1]...] -> [4,5,...]
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)

    # 获取有效token的一维索引
    # attention_mask展成一维，然后获取有效位置
    # [1,1,1,1,0,1,1,1,1,1,...] -> [0,1,2,3,5,6,7,8,9,...]
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    # 批次中最大序列长度
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())

    # 计算累计序列长度，最前面补一个0
    # [4,5,...] -> [0,4,9,...]
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))

    # 将hidden_states展平到(batch * seqlen, ...)的形状，取出有效token
    hidden_states = cast(torch.Tensor, index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices))

    return hidden_states, indices, cu_seqlens, max_seqlen_in_batch

# modernBERT中的，参考
def unpad_input_only_refer(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """只返回去除padding后的hidden_states，节省少量开销。

    参数:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen)

    返回:
        hidden_states: (total_nnz, ...)
    """
    # 获取有效token索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()

    # 展平hidden_states
    rearranged = rearrange(hidden_states, "b s ... -> (b s) ...")

    # 返回索引后的结果
    return index_first_axis(rearranged, indices)  # type: ignore

def unpad_input(
    hidden_states: torch.Tensor,      # (batch, seqlen, dim)
    sample_ids_seq: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """移除输入序列中的padding。

    参数:
        hidden_states: (batch, seqlen, ...)
        sample_ids_seq: List of length B, each is a 1D LongTensor of shape (L_i,)
                        L_i ≤ S，表示这条序列前 L_i 个 token 有效，后面都是 padding。
                        每个元素值 ∈ {0,1,…,N_i−1}，标记属于哪个样本。

    返回:
        hidden_states: (total_nnz, ...) ，去除padding后的张量
        indices: (total_nnz)，有效token的位置
        cu_seqlens: (batch + 1)，样本累计长度
        max_seqlen: 最大样本长度
    """
    bs, seq_len = hidden_states.shape[:2]
    device = hidden_states.device

    # 构造 mask
    mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=device)
    for i, ids in enumerate(sample_ids_seq):
        L = ids.numel()
        assert L <= seq_len, f"第 {i} 条序列长度 {L} 超过 seqlen={seq_len}"
        mask[i, :L] = True

    # 获取有效token的一维索引(和之前相同)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    # 将hidden_states展平到(batch * seqlen, ...)的形状，取出有效token(和之前相同)
    hidden_states = cast(torch.Tensor, index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices))

    # 计算每个样本的长度列表
    lengths = []
    for i, ids in enumerate(sample_ids_seq):
        n_sub = int(ids.max().item()) + 1
        # 对每个子样本，统计有多少 token
        for s in range(n_sub):
            cnt = int((ids == s).sum().item())
            lengths.append(cnt)

    # 构造cu_seqlens和max_seqlen
    lens_tensor = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu_seqlens = torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.cumsum(lens_tensor, dim=0, dtype=torch.int32)
    ], dim=0)

    max_seqlen = int(lens_tensor.max().item())

    return hidden_states, indices, cu_seqlens, max_seqlen

# 仅移除padding，只返回hidden_states
def unpad_input_only(
    hidden_states: torch.Tensor,
    sample_ids_seq: list[torch.Tensor]
) -> torch.Tensor:
    """只返回去除padding后的hidden_states，节省开销。

    参数:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen)

    返回:
        hidden_states: (total_nnz, ...)
    """
    bs, seq_len = hidden_states.shape[:2]
    device = hidden_states.device

    # 构造 mask
    mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=device)
    for i, ids in enumerate(sample_ids_seq):
        L = ids.numel()
        assert L <= seq_len, f"第 {i} 条序列长度 {L} 超过 seqlen={seq_len}"
        mask[i, :L] = True
    # 获取有效token的一维索引(和之前相同)
    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    # 展平hidden_states
    rearranged = rearrange(hidden_states, "b s ... -> (b s) ...")
    # 返回索引后的结果
    return index_first_axis(rearranged, indices)

# 添加padding，将unpad后的hidden_states恢复原形
def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """为序列添加padding。

    参数:
        hidden_states: (total_nnz, ...) ，无padding的隐藏状态，其中total_nnz = attention_mask中选中的token数.
        indices: (total_nnz)，有效token的位置
        batch: 批次大小
        seqlen: 序列长度

    返回:
        hidden_states: (batch, seqlen, ...)
    """
    # 将hidden_states根据indices放入(batch * seqlen, ...)的张量中
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)

    # 恢复成(batch, seqlen, ...)的形状
    return rearrange(output, "(b s) ... -> b s ...", b=batch)