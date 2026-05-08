import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.modeling_outputs import ImageClassifierOutput
from transformers.utils import ModelOutput

# 通过@dataclass装饰之后不需要__init__方法
# Gabrielle 0314：自定义多任务输出
@dataclass
class MultiTaskOutput(ImageClassifierOutput):
    """
    Output class for multi-task models with classification and regression.
    Inherits from ImageClassifierOutput and The logits is a dictionary containing the prediction values for classification and regression tasks..
    """
    loss: torch.Tensor
    logits: dict
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None
    # gwj 0526: 加入分任务的loss_details
    loss_details: Optional[dict] = None  # 分任务的loss详情
    
@dataclass
class SiTMAEModelOutput(ModelOutput):
    """
    Class for SiTMAEModel's outputs, with potential hidden states and attentions.
    SiTMAEModel输出的类，包括隐藏状态和注意力。

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None  # 模型最后一层输出的隐藏状态 (bs, max_patches_len, hidden_size)
    mae_mask: torch.LongTensor = None  # Gabrielle: 改名
    ids_restore: torch.LongTensor = None  # 恢复索引
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型在每一层输出的隐藏状态加上初始嵌入输出
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SiTMAEDecoderOutput(ModelOutput):
    """
    Class for SiTMAEDecoder's outputs, with potential hidden states and attentions.
    SiTMAE解码器输出的类。
    """
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class SiTMAEForPreTrainingOutput(ModelOutput):
    """
    Class for SiTMAEForPreTraining's outputs, with potential hidden states and attentions.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`):
            Pixel reconstruction loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, patch_size ** 2 * num_channels)`):
            Pixel reconstruction logits.
        mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Tensor indicating which patches are masked (1) and which are not (0).
        ids_restore (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Tensor containing the original index of the (shuffled) masked patches.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    attn_mask: torch.LongTensor = None
    ids_restore: torch.LongTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None