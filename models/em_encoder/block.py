import torch
import math
from transformers.activations import ACT2FN
from .configuration_em import SiTMAEConfig
from torch import nn
from typing import Optional, Tuple, Union
# zzy 0502 导入fa库
import importlib.metadata
IMPL_USE_FLASH2 = False
try:
    from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func

    installed_version = importlib.metadata.version("flash_attn")
    if installed_version < "2.5.7":
        raise ImportError("newer version of flash_attn required (>= 2.5.7)")
    IMPL_USE_FLASH2 = True
except ImportError:
    pass

class BaseSiTMAEAttention(nn.Module):
    """共有逻辑"""
    def __init__(self, config: SiTMAEConfig) -> None:
        """初始化注意力层参数"""
        super().__init__()
        # 检查隐藏层大小是否能被注意力头数整除
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        # 基础参数设置
        self.num_attention_heads = config.num_attention_heads  # 注意力头数量
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 每个头的维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 所有头的总维度
        # 定义Q/K/V线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        # 注意力概率的dropout层
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 原本SiTMAESelfOutput中的逻辑，一个全连接层，一个dropout
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """将输入张量重塑为多头注意力需要的形状
        Args:
            x: 输入张量 shape(batch_size, max_patches_len, hidden_size)
        Returns:
            重塑后的张量 shape(batch_size, num_heads, max_patches_len, head_size)
        """
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)  # 重塑张量形状
        return x.permute(0, 2, 1, 3)  # 调整维度顺序

class SiTMAEAttention(BaseSiTMAEAttention):
    """实现基于多头注意力机制的Self-Attention层，支持注意力掩码和头部掩码"""
    def __init__(self, config: SiTMAEConfig) -> None:
        super().__init__(config)  # 继承父类初始化

    def forward(
            self, 
            hidden_states, 
            head_mask: Optional[torch.Tensor] = None, 
            output_attentions: bool = False,
            mask: Optional[torch.LongTensor] = None,  # 注意力掩码，形状(batch_size, max_patches_len, max_patches_len)
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        Args:
            hidden_states: 输入隐藏状态 shape(batch_size, max_patches_len, hidden_size)
            head_mask: 头部掩码，可选
            output_attentions: 是否输出注意力权重
            mask: 注意力掩码，True表示需要被mask的位置
        Returns:
            注意力输出结果，可选包含注意力权重
        """
        # 计算Q/K/V
        mixed_query_layer = self.query(hidden_states)  # 计算查询向量
        # 转置K/V/Q为多头格式
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # 计算原始注意力分数
        # 前两维（batch_size和num_heads）保持独立，即按元素逐批、逐头计算
        # (batch_size, num_heads, max_patches_len_k, head_size)->(batch_size, num_heads, max_patches_len_k, max_patches_len_k)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 缩放注意力分数
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 处理注意力掩码（Gabrielle 0325添加的预处理步骤）
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)  # 扩展掩码维度(b,12,258,258)
            attention_scores = attention_scores.masked_fill(~mask, -1e4)        # 用极小值填充被掩码位置
        # 计算注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # 应用dropout（原Transformer论文中的设计）
        attention_probs = self.attention_dropout(attention_probs)
        # 应用头部掩码（如果提供）
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        # 计算上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        # 合并多头输出
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        # context_layer传入原本是SiTMAESelfOutput中的组件
        # 这里未实现残差连接
        context_layer = self.dropout(self.dense(context_layer))
        # context_layer作为实际的特征表示，传递给后续网络层
        # attention_probs提供模型决策过程的透明度（可解释性），适合可视化分析或特定任务（如关系抽取）
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SiTMAESdpaAttention(BaseSiTMAEAttention):
    """基于PyTorch原生scaled_dot_product_attention实现的高效自注意力层，继承自SiTMAESelfAttention"""
    
    def __init__(self, config: SiTMAEConfig) -> None:
        """初始化，继承父类参数并获取dropout概率"""
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob  # 保存dropout概率

    def forward(
            self, 
            hidden_states, 
            head_mask: Optional[torch.Tensor] = None, 
            output_attentions: bool = False,
            mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        """
        参数:
            hidden_states: 输入隐藏状态 shape(batch_size, max_patches_len, hidden_size)
            head_mask: 头部掩码（未使用，为API兼容保留）
            output_attentions: 是否输出注意力权重（强制返回None以兼容父类）
            mask: 注意力掩码 shape(batch_size, max_patches_len, max_patches_len)
        
        返回:
            Tuple[上下文张量, None]（为保持接口统一）
        """
        # 计算查询向量（Q）
        mixed_query_layer = self.query(hidden_states)

        # 计算并转置键（K）、值（V）、查询（Q）为多头格式
        key_layer = self.transpose_for_scores(self.key(hidden_states))  # (batch_size, num_heads, max_patches_len, head_size)
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 初始化掩码（当前版本未实际处理mask）
        mask_for_sdpa = None

        mask_for_sdpa = mask_for_sdpa.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)  # 无效果（None仍为None）

        # 使用PyTorch优化后的scaled_dot_product_attention计算注意力
        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,  # 查询张量
            key_layer,    # 键张量
            value_layer,  # 值张量
            attn_mask=mask_for_sdpa,  # 注意力掩码（实际未使用）
            dropout_p=self.attention_probs_dropout_prob if self.training else 0.0,  # 训练时启用dropout
            is_causal=False,  # 非因果注意力（适用于编码器）
            scale=None,       # 使用默认缩放因子sqrt(head_size)
        )
        # 合并多头输出
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # (bs, max_patches_len, heads, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # 合并最后两维
        context_layer = context_layer.view(new_context_layer_shape)  # (bs, max_patches_len, hidden_size)
        # context_layer传入原本是SiTMAESelfOutput中的组件
        context_layer = self.dropout(self.dense(context_layer))
        # 返回上下文张量
        return context_layer

# zzy 0513 使用fa的注意力层
class SiTMAEUnpadFlashAttention(nn.Module):
    def __init__(self, config: SiTMAEConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.num_attention_heads = config.num_attention_heads  # 注意力头数
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 每个头维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.Wqkv = nn.Linear(config.hidden_size, 3 * self.all_head_size, bias=config.qkv_bias)  # Q/K/V线性变换层
        self.attention_dropout = config.attention_probs_dropout_prob  # 注意力的dropout概率
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,    # (total_nnz, dim)
        cu_seqlens: torch.Tensor,       # (batch + 1,)
        max_seqlen: int,                # int
    ):
        bs, dim = hidden_states.shape
        qkv = self.Wqkv(hidden_states)
        if IMPL_USE_FLASH2:
            # 需要的形状(total, 3, nheads, headdim)
            qkv = qkv.view(-1, 3, self.num_attention_heads, self.attention_head_size)
            convert_dtype = qkv.dtype not in (torch.float16, torch.bfloat16)
            if convert_dtype:
                orig_dtype = qkv.dtype
                qkv = qkv.to(torch.bfloat16)
                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.attention_dropout
                )
                attn = attn.to(orig_dtype)
            else:
                attn = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    dropout_p=self.attention_dropout
                )
            attn = attn.view(bs, dim)
        # 与之前的返回值保持一致，返回元组
        return (self.dropout(self.Wo(attn)), )

# zzy 0513
# 注意力类的映射，自己实现的注意力层；调用库函数实现的注意力层；使用fa的注意力层
SITMAE_ATTENTION_CLASSES = {
    "eager": SiTMAEAttention,
    "sdpa": SiTMAESdpaAttention,
    "fa2": SiTMAEUnpadFlashAttention
}

class SiTMAELayer(nn.Module):
    """
    This corresponds to the Block class in the timm implementation.
    SiTMAE模型的基础层模块，对应timm实现中的Block类。
    包含完整的自注意力机制和前馈网络，以及两次残差连接和层归一化操作。
    """
    def __init__(self, config: SiTMAEConfig) -> None:
        super().__init__()
        # zzy 0527 使用不同注意力层的逻辑
        self.attention = SITMAE_ATTENTION_CLASSES[config.attention_type](config)
        if config.attention_type == "fa2":
            if IMPL_USE_FLASH2:
                self.use_fa2 = True
            else:
                raise ValueError("No flash_attn required environment")
        else:
            self.use_fa2 = False
        # 两个layer norm
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义FFN FFN(x) = act(xW_1 + b_1)W_2 + b_2
        self.w_1 = nn.Linear(config.hidden_size, config.intermediate_size)  # 升维
        if isinstance(config.hidden_act, str):  # 如果hidden_act是字符串，则从预定义的ACT2FN映射中获取对应的激活函数
            self.act_fn = ACT2FN[config.hidden_act]
        else:  # 如果hidden_act已经是函数对象，则直接使用
            self.act_fn = config.hidden_act  # gelu
        self.w_2 = nn.Linear(config.intermediate_size, config.hidden_size)  # 降维
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # dropout层防止过拟合

    def forward(
            self,
            hidden_states: torch.Tensor,  # 输入隐藏状态
            head_mask: Optional[torch.Tensor] = None,  # 注意力头掩码，用于屏蔽特定注意力头
            output_attentions: bool = False,  # 是否输出注意力权重
            mask: Optional[torch.LongTensor] = None,  # 编码器注意力掩码（encoder_attn_mask）
            # zzy 0513 用于fa
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # Pre-LN流程
        # 自注意力计算
        # zzy 0513 两个注意力层的参数不同
        if self.use_fa2:
            self_attention_outputs = self.attention(
                self.norm1(hidden_states),  # (total, hidden_size)
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen
            )
        else:
            self_attention_outputs = self.attention(
                self.norm1(hidden_states),  # (bs, max_patches_len, hidden_size) 自注意力前层归一化
                head_mask=head_mask,
                output_attentions=output_attentions,
                mask=mask,  # 编码器注意力掩码
            )
        attention_output = self_attention_outputs[0]  # 获取自注意力输出
        outputs = self_attention_outputs[1:]  # 如果需要，保存注意力权重
        # 第一次残差连接
        hidden_states = attention_output + hidden_states
        # 自注意力后的层归一化；前馈层传播；第二次残差连接
        layer_output = hidden_states + self.dropout(self.w_2(self.act_fn(self.w_1(self.norm2(hidden_states)))))
        # 如果需要输出注意力权重，将其添加到输出中
        outputs = (layer_output,) + outputs
        return outputs