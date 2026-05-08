# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


import math
import collections.abc
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

# Embedding Layers for Timesteps and Condition Inputs
class TimestepEmbedder(nn.Module):
    """
    将时间步长嵌入向量表示中
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, frequency_embedding_size, max_period=10000):
        """
        创建正弦时间步长嵌入
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element. These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = frequency_embedding_size // 2
        # 生成一列递减的数，10000^(-i/half)，i in [0,half-1]
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        # t[:, None]将t变为(N, 1)，freqs[None]将freqs变为(1, half)
        # 相乘时进行广播
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # dim是奇数则补全到dim的长度
        if frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        # t.shape = (N) -> t_freq.shape = (N, frequency_embedding_size=256)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # t_emb.shape = (N, hidden_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimestepEmbedder_trainable(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype
        
        # 将频率设为可训练参数，初始化为 10000^(2i/d)
        half = frequency_embedding_size // 2
        init_freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
        self.freqs = nn.Parameter(init_freqs, requires_grad=True)

    def timestep_embedding(self, t, frequency_embedding_size):
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimestepEmbedder_triple(nn.Module):
    """
    不同单位制下的TimestepEmbedder
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size*3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    def timestep_embedding(self, t, frequency_embedding_size, max_period=10000):
        half = frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding.to(self.dtype)

    def forward(self, t):
        t_freq_k = self.timestep_embedding(t/1000, self.frequency_embedding_size)
        t_freq_m = self.timestep_embedding(t/1000000, self.frequency_embedding_size)
        t_freq_g = self.timestep_embedding(t/1000000000, self.frequency_embedding_size)
        t_emb = self.mlp(torch.cat([t_freq_k, t_freq_m, t_freq_g], dim=-1))
        return t_emb

class SiTMAEEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    构建CLS token、位置信息和patch的嵌入。
    """
    def __init__(self, config):
        super().__init__()
        # Gabrielle 0516: 根据 cls 和 fs 增大 max_seq_len
        max_seq_len = config.max_seq_len[0] + (16 if config.use_fs else 8)
        max_seq_len = max_seq_len if isinstance(max_seq_len, collections.abc.Iterable) else (max_seq_len, max_seq_len)

        # 确保patch_size是包含两个元素的迭代对象（如元组），如果本身就是则使用原值
        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (config.patch_size, config.patch_size)
        # 一个seq最多可以划分为多少patch
        self.num_patches = max_seq_len[0] // patch_size[0]
        # 一个patch的维度 C*H*W
        patch_dim = config.num_channels * patch_size[0] * patch_size[1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.fs_embedder = TimestepEmbedder(config.hidden_size, dtype=torch.float32)
        self.position_embeddings = nn.Parameter(torch.randn(self.num_patches, config.hidden_size))
        # patch_dim -> hidden_size的映射
        self.projection = nn.Linear(patch_dim, config.hidden_size)

        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        w = self.position_embeddings.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)
        # 初始化TimestepEmbedder中mlp的weight
        nn.init.normal_(self.fs_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.fs_embedder.mlp[2].weight, std=0.02)

    # zzy 0429: gen_attn_mask改为类内静态方法
    @staticmethod
    def gen_attn_mask(sample_ids_seq, max_patches_len):
        '''
        根据sample_ids_seq生成attn_mask
        sample_ids_seq形式为list[tensor] [(0,0,0,0,1,1,1), (0,0,1,1,2,2,0)]
        '''
        seq_arange = torch.arange(max_patches_len)
        # 每个seq的真实长度
        lengths = torch.tensor([p.shape[0] for p in sample_ids_seq], dtype = torch.int)
        # 用0补全到最大长度
        sample_ids_seq = torch.stack([F.pad(p, (0, max_patches_len - len(p)), value=0) for p in sample_ids_seq], dim=0)
        # 标记哪些位置是有效的，padding的位置为False
        key_pad_mask = (rearrange(seq_arange, 'n -> 1 n') < rearrange(lengths, 'b -> b 1')).to(sample_ids_seq.device)
        # 只看信号 ID 是否相同，决定哪些patch可以互相关注
        attn_mask = rearrange(sample_ids_seq, 'b i -> b i 1') == rearrange(sample_ids_seq, 'b j -> b 1 j')
        # 用key_pad_mask双向过滤，确保只有真实patch互相关注，填充部分被屏蔽
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 j') & rearrange(key_pad_mask, 'b j -> b j 1')
        return attn_mask

    def shuffle_in_sample(self, tensor, start_indices):
        '''
        打乱sample内部的噪声
        start_indices.shape=(bs, max_patches_len)
        '''
        prefix_len = 2 if self.config.use_fs else 1
        # 将tensor转换为列表以便逐行操作
        tensor_list = tensor.tolist()
        # 遍历tensor每一行及其对应的start_indices
        for i in range(tensor.size(0)):
            row = tensor_list[i]
            row_starts = start_indices[i]
            # 遍历每个sample，进行打乱（除最后一个）
            for i in range(len(row_starts) - 1):  
                start = row_starts[i] + prefix_len  # 根据需求，前prefix个不打乱
                end = row_starts[i + 1]
                sample = row[start:end]  # 提取当前sample的数据
                perm = torch.randperm(end - start)  # 生成随机排列的索引
                shuffled_sample = [sample[idx] for idx in perm.tolist()]  # 打乱sample内部数据
                row[start:end] = shuffled_sample  # 放回原处
        # 将列表转换回tensor
        return torch.tensor(tensor_list).to(tensor.device)

    def random_masking_strict_ratio(self, sample_embeddings, sample_ids_seq):
        '''
        sample_embeddings.shape = (bs, max_patches_len, hidden_size), 
        sample_ids_seq list[tensor], 标识各样本，形式为[(0,0,0,1,1,2), (0,0,0,0,1,1,1)]
        控制每个样本的cls token和fs被取到，iq数据取1-mask_ratio
        '''
        device = sample_embeddings.device
        bs, max_patches_len, dim = sample_embeddings.shape
        
        # 计算各样本长度
        lengths_list = []
        for p in sample_ids_seq:
            # bincount用于计算一维整型张量每个非负数出现的次数
            # 返回一个一维张量 shape=(max(input)+1)，索引i处的值表示i在输入张量中出现的频次
            lengths = torch.bincount(p)
            lengths_list.append(lengths)  # [3, 2, 4]

        # 计算各样本起始索引
        start_indices = [] 
        for lengths in lengths_list:
            start_index = torch.cumsum(lengths[:], dim=0) # cumsum累加计算，每个元素等于之前所有元素之和，在这里对应每段的起始索引
            start_indices.append(torch.cat((torch.tensor([0], device=device), start_index), dim=0))
         
        # 生成噪声
        simulate_noise = []
        prefix_len = 1 if self.config.use_fs else 0  # 如果数据包含fs，前面补一个0，否则不需要
        for lengths in lengths_list:
            segments = []
            for l in lengths:
                l = l.item()
                # 对于每个样本长度l：若不包含fs，前面1个0，后面的数为 1/(l-1), 2/(l-1), ..., 1
                # 若包含fs，前面2个0，后面的数为 1/(L-2), 2/(L-2), ..., 1
                seg = torch.cat([torch.zeros(prefix_len), torch.linspace(0, 1, steps=l-prefix_len)])
                segments.append(seg)
            # 拼接该行所有段
            row = torch.cat(segments)
            # 若不足 max_patches_len，用 inf 补全
            if row.numel() < max_patches_len:
                pad = torch.full((max_patches_len - row.numel(),), float('inf'))
                row = torch.cat([row, pad])
            simulate_noise.append(row)
        simulate_noise = torch.stack(simulate_noise).to(sample_embeddings.device)
        # 逐样本打乱
        simulate_noise = self.shuffle_in_sample(simulate_noise, start_indices)

        # 计算len_keep
        num_samples = [len(tensor) for tensor in lengths_list]  # 每行有几个样本 [3]
        num_samples = torch.tensor(num_samples, device=device)
        effective_lens = [torch.sum(tensor, dim=0) for tensor in lengths_list]  # 每行有效数据的长度
        effective_lens = torch.tensor(effective_lens, device=device)  # [9]
        # 每一行要保留的长度（cls token和fs必须保留，iq数据取1-mask_radio）
        len_keep = 2*num_samples + ((effective_lens-2*num_samples)*(1-self.config.mask_ratio)).int()
        len_keep_max, _ = len_keep.max(dim=0)

        # shuffle 升序排列保证所有的 cls/fs token 都保留（因为它们都是0）
        ids_shuffle = torch.argsort(simulate_noise, dim=1)
        # argsort获取噪声排序后的索引，会将样本间的token混在一起，需要将前len_keep个再次排序
        # 然后再进行gather，使得seq_unmasked中仍然是按照样本排序的
        for i in range(bs):
            k = len_keep[i]
            ids_shuffle[i, :k], _ = torch.sort(ids_shuffle[i, :k]) # 对保留部分重新排序，模型可见这些 token
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 对每一行进行掩码
        ids_keep_list, sequence_unmasked_list = [], []
        for i in range(bs):
            row_ids_keep = ids_shuffle[i][:len_keep[i]]
            row_unmasked = torch.gather(sample_embeddings[i], dim=0, index=row_ids_keep.unsqueeze(-1).repeat(1, dim))
            ids_keep_list.append(row_ids_keep)
            sequence_unmasked_list.append(row_unmasked)
        
        # 对 sequence_unmasked_list padding 并转化为 tensor, 对应 mae_mask = 0
        sequence_unmasked = pad_sequence(sequence_unmasked_list, batch_first=True, padding_value=0)
        
        # 让掩码第i行的前len_keep[i]个元素为0
        mae_mask = torch.ones([bs, max_patches_len], device=device) # 1 表示掩码掉
        indices = torch.arange(max_patches_len, device=device).unsqueeze(0).repeat(bs, 1)

        mae_mask[indices < len_keep.unsqueeze(1)] = 0 # 前 len_keep 设为0，表示不计算损失，模型可见这些 token
        # zzy 0520 上次改得不对，这里应该包含等号
        mae_mask[indices >= effective_lens.unsqueeze(1)]= 0 
        mae_mask = torch.gather(mae_mask, dim=1, index=ids_restore)  # unshuffle

        # zzy 0527 如果使用fa，不需要生成encoder_attn_mask，但是要生成一个新的sample_ids_seq，表示保留部分的长度
        if self.config.attention_type == "fa2":
            encoder_attn_mask = None
            unmask_sample_ids_seq = []
            for i in range(bs):
                temp = torch.gather(input=sample_ids_seq[i], dim=0, index=ids_keep_list[i])
                unmask_sample_ids_seq.append(temp)
        else:
            # 生成attn_mask
            attn_mask = self.gen_attn_mask(sample_ids_seq, max_patches_len)
            # 生成encoder_attn_mask，encoder_attn_mask[i]指示sequence_unmasked[i]中互相关注（出自同一个sample）的patch
            encoder_attn_mask = torch.zeros(
                                    (bs, len_keep_max, len_keep_max), 
                                    dtype=torch.bool, 
                                    device=device
                                )
            for i in range(bs):
                # 从 attn_mask 中提取保留的 token 对应的子矩阵
                sub_mask = attn_mask[i][ids_keep_list[i]][:, ids_keep_list[i]]  # 提取子矩阵
                # 注意 sub_mask 这时的 size 可能小于 encoder_attn_mask
                encoder_attn_mask[i][:len_keep[i], :len_keep[i]] = sub_mask.bool()  # 转换为布尔类型
            unmask_sample_ids_seq = None
        
        # sequence_unmasked 保留的部分 (bs, len_keep_max, hidden_size)
        # ids_restore 恢复索引 (bs, max_patches_len)
        # mae_mask 掩码 (bs, max_patches_len)
        # encoder_attn_mask (bs, len_keep_max, len_keep_max)
        return sequence_unmasked, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_ids_seq

    def random_masking_row(self, sample_embeddings, sample_ids_seq):
        '''
        所有样本的cls和fs被取到，不控制每个样本的比例，每行的iq数据取1-mask_ratio
        '''
        device = sample_embeddings.device
        bs, max_patches_len, dim = sample_embeddings.shape
        # 计算各样本长度
        lengths_list = []
        for p in sample_ids_seq:
            lengths = torch.bincount(p)
            lengths_list.append(lengths)
        # 计算各样本起始索引
        start_indices = [] 
        for lengths in lengths_list:
            start_index = torch.cumsum(lengths[:-1], dim=0)  # 这里扔掉最后一个
            start_indices.append(torch.cat((torch.tensor([0], device=device), start_index), dim=0))
        # 计算len_keep
        num_samples = [len(tensor) for tensor in lengths_list]
        num_samples = torch.tensor(num_samples, device=device)
        effective_lens = [torch.sum(tensor, dim=0) for tensor in lengths_list]
        effective_lens = torch.tensor(effective_lens, device=device)
        len_keep = 2*num_samples + ((effective_lens-2*num_samples)*(1-self.config.mask_ratio)).int()
        len_keep_max, _ = len_keep.max(dim=0)
        # 生成噪声
        noise = torch.rand(bs, max_patches_len, device=device)  # noise in [0, 1]
        cls_indices = torch.cat([
            torch.stack([torch.full_like(t, i), t], dim=1)
            for i, t in enumerate(start_indices)
        ], dim=0)
        noise[cls_indices[:, 0], cls_indices[:, 1]] = 0
        if self.config.use_fs:
            next_col = cls_indices[:, 1] + 1
            noise[cls_indices[:, 0], next_col] = 0
        mask = torch.arange(noise.size(1), device=noise.device).unsqueeze(0) >= effective_lens.unsqueeze(1)
        noise = noise.masked_fill(mask, float('inf'))
        # shuffle
        ids_shuffle = torch.argsort(noise, dim=1)
        for i in range(bs):
            k = len_keep[i]
            ids_shuffle[i, :k], _ = torch.sort(ids_shuffle[i, :k])
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # 对每一行进行掩码
        ids_keep_list, sequence_unmasked_list = [], []
        for i in range(len(sample_embeddings)):
            row_ids_keep = ids_shuffle[i][:len_keep[i]]
            row_unmasked = torch.gather(sample_embeddings[i], dim=0, index=row_ids_keep.unsqueeze(-1).repeat(1, dim))
            ids_keep_list.append(row_ids_keep)
            sequence_unmasked_list.append(row_unmasked)
        # 对sequence_unmasked_list padding并转化为tensor
        sequence_unmasked = pad_sequence(sequence_unmasked_list, batch_first=True, padding_value=0)
        # 生成mae_mask
        mae_mask = torch.ones([bs, max_patches_len], device=device)
        indices = torch.arange(max_patches_len, device=device).unsqueeze(0).repeat(bs, 1)
        mae_mask[indices < len_keep.unsqueeze(1)] = 0
        # zzy 0520
        mae_mask[indices >= effective_lens.unsqueeze(1)]= 0  # Gabrielle 0520: 修改 mae_mask 将 padding 部分也设为0
        mae_mask = torch.gather(mae_mask, dim=1, index=ids_restore)

        # zzy 0527 如果使用fa，不生成encoder_attn_mask
        if self.config.attention_type == "fa2":
            encoder_attn_mask = None
            unmask_sample_ids_seq = []
            for i in range(bs):
                temp = torch.gather(input=sample_ids_seq[i], dim=0, index=ids_keep_list[i])
                unmask_sample_ids_seq.append(temp)
        else:
            attn_mask = self.gen_attn_mask(sample_ids_seq, max_patches_len)
            encoder_attn_mask = torch.zeros(
                                    (bs, len_keep_max, len_keep_max), 
                                    dtype=torch.bool,
                                    device=sample_embeddings.device
                                )
            for i in range(bs):
                sub_mask = attn_mask[i][ids_keep_list[i]][:, ids_keep_list[i]]
                encoder_attn_mask[i][:len_keep[i], :len_keep[i]] = sub_mask.bool()

        return sequence_unmasked, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_ids_seq

    def forward(self,
                patches: torch.Tensor,          # (bs, max_patches_len, patch_dim) (b, 18, 16)
                patch_positions: torch.Tensor,  # patch_positions (bs, max_patches_len)
                sample_ids_seq: List[torch.Tensor],
                noise: Optional[torch.Tensor] = None,
                enable_mae_mask: bool = True,
                ):
        # 在投影到768维之前先记录fs，要不然会变
        if self.config.use_fs:
            fs_mask = (patch_positions == 1)
            selected_patch = patches[fs_mask]  # (num_selected, patch_dim)
            fs = selected_patch[:, 0]  # (num_selected,)
        patch_embeddings = self.projection(patches)  # (bs, max_patches_len, 16) -> (bs, max_patches_len, 768)
        bs, max_patches_len, hidden_size = patch_embeddings.shape
        # 指示cls token的位置
        cls_mask = (patch_positions == 0).unsqueeze(-1)  # (bs, max_patches_len, 1)
        # 扩展cls_token到匹配patch的维度
        cls_token_expanded = self.cls_token.expand(bs, max_patches_len, hidden_size)
        # where(condition, x, y) 将相应位置替换为cls token
        patch_embeddings = torch.where(cls_mask, cls_token_expanded, patch_embeddings)
        # 如果包含频率
        if self.config.use_fs:
            # 频率送进fs_embedder
            # 由于开启混合精度计算，虽然fs_embedder的参数是float32，中间结果的数据类型会被转成bf16
            fs_embeddings = self.fs_embedder(fs).to(patch_embeddings.dtype)
            patch_embeddings[fs_mask] = fs_embeddings  # 覆盖原始fs位置
        # 添加位置编码
        patch_embeddings = patch_embeddings + self.position_embeddings[patch_positions]

        # zzy 0520 加一个返回值
        if enable_mae_mask:
            embeddings, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_eds_seq = self.random_masking_strict_ratio(
                patch_embeddings, sample_ids_seq
                )
        else:
            # zzy 0520 修改逻辑，也分为是否使用fa
            embeddings = patch_embeddings
            ids_restore = None
            mae_mask = None
            # zzy 0527
            if self.config.attention_type == "fa2":
                unmask_sample_eds_seq = sample_ids_seq  # 不掩码时，sample_ids_seq不用处理
                encoder_attn_mask = None
            else:
                attn_mask = self.gen_attn_mask(sample_ids_seq, max_patches_len)
                encoder_attn_mask = attn_mask  # 不掩码时，encoder_attn_mask就是attn_mask
                unmask_sample_eds_seq = None
        # zzy 0520 返回值
        return embeddings, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_eds_seq