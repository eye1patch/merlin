from copy import deepcopy
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, L1Loss

from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

# zzy 0513 修改导入
from .configuration_em import SiTMAEConfig
from .embedder import SiTMAEEmbeddings
from .dataclass import *
from .block import SiTMAELayer
from .padding import *

logger = logging.get_logger(__name__)

class SiTMAEEncoder(nn.Module):
    """SiTMAE模型的编码器模块，由多个SiTMAELayer块堆叠组成"""

    def __init__(self, config: SiTMAEConfig) -> None:
        super().__init__()
        self.config = config
        # num_hidden_layers 模型层数
        self.layer = nn.ModuleList(
            [SiTMAELayer(config) for _ in range(config.num_hidden_layers)]
        )
        # 梯度检查点标志，默认为False
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,  # (batch_size, max_patches_len, hidden_size)
            head_mask: Optional[torch.Tensor] = None,  # 注意力头掩码，(num_layers, num_heads)
            output_attentions: bool = False,  # 是否输出各层的注意力权重
            output_hidden_states: bool = False,  # 是否输出各层的隐藏状态
            return_dict: bool = True,  # 是否以字典形式返回结果
            mask: Optional[torch.LongTensor] = None,  # encoder_attn_mask
            # zzy 0513 用于fa
            cu_seqlens: Optional[torch.Tensor] = None,
            max_seqlen: Optional[int] = None
    ) -> Union[tuple, BaseModelOutput]:
        # 初始化用于收集各层隐藏状态和注意力权重的容器
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # 逐层处理
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，保存当前层的输入
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # 获取当前层的注意力头掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 梯度检查点技术（在训练时节省显存）
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                    mask=mask,  # encoder_attn_mask
                    # zzy 0513
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                )
            else:
                # 前向传播
                layer_outputs = layer_module(
                    hidden_states, 
                    layer_head_mask, 
                    output_attentions,
                    mask=mask,  # encoder_attn_mask
                    # zzy 0513
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                )
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力权重，保存当前层的注意力权重
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        # 如果需要输出隐藏状态，保存最后一层的输出
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        # 根据return_dict参数决定返回格式
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,  # 最终层的输出
            hidden_states=all_hidden_states,  # 所有层的隐藏状态（如果output_hidden_states为True）
            attentions=all_self_attentions,  # 所有层的注意力权重（如果output_attentions为True）
        )


class SiTMAEPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    一个用于处理权重初始化的抽象类和一个用于下载和加载预训练模型的接口
    子类通过 super().__init__(config) 调用父类初始化逻辑，触发 _init_weights 方法，使所有子模块正确初始化
    """

    config_class = SiTMAEConfig
    base_model_prefix = "sit"  # 模型前缀
    main_input_name = "patches"  # 模型的主要输入是图像块(patches)
    supports_gradient_checkpointing = True  # 支持梯度检查点技术
    _supports_sdpa = True  # 支持缩放点积注意力(scaled dot-product attention)的高效实现

    def _init_weights(self, module):
        """Initialize the weights"""
        # 对线性层(nn.Linear)和卷积层(nn.Conv2d)：按正态分布初始化weight，bias初始为0
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        # 对层归一化(nn.LayerNorm)：weight初始化为1，bias初始化为0
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SiTMAEModel(SiTMAEPreTrainedModel):
    """SiTMAE模型的主类，包含完整的嵌入层、编码器和输出处理"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 嵌入层（处理patch嵌入和位置编码）
        self.embeddings = SiTMAEEmbeddings(config)
        # 编码器（多层Transformer结构）
        self.encoder = SiTMAEEncoder(config)
        # 最终层归一化
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化权重并应用最终处理
        self.post_init()

    def forward(
        self,
        patches: torch.FloatTensor,
        patch_positions: torch.FloatTensor,  # patch位置信息
        sample_ids_seq: list[torch.Tensor],
        noise: Optional[torch.FloatTensor] = None,  # 噪声输入（用于MAE）
        head_mask: Optional[torch.FloatTensor] = None,  # 注意力头掩码
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式
    ) -> Union[Tuple, SiTMAEModelOutput]:
        # zzy 0520
        bs, seq_len, _ = patches.shape
        # 设置默认输出选项（这三个config都没定义）
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # 输入验证
        if patches is None:
            raise ValueError("You have to specify patches")
        # 处理head_mask（1表示保留该注意力头）
        # 将head_mask转换为合适的形状 (num_hidden_layers, batch, num_heads, max_patches_len, max_patches_len)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        # 通过嵌入层获取嵌入表示
        # 返回：嵌入之后保留的部分、恢复ID、MAE掩码、编码器注意力掩码
        # zzy 0520 增加unmask_sample_ids_seq，即保留部分样本的长度标记
        patches, ids_restore, mae_mask, encoder_attn_mask, unmask_sample_ids_seq = self.embeddings(
            patches,
            patch_positions=patch_positions,   # 传入patch位置信息
            sample_ids_seq = sample_ids_seq,
            noise=noise,
        )
        # zzy 0527 在嵌入之后将patch展开为没有padding的超长序列
        if self.config.attention_type == "fa2":
            patches, indices, cu_seqlens, max_seqlen = unpad_input(patches, unmask_sample_ids_seq)
        else:
            cu_seqlens, max_seqlen = None, None

        # 通过编码器处理嵌入表示
        encoder_outputs = self.encoder(
            patches,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask=encoder_attn_mask,  # 使用编码器注意力掩码
             # zzy 0513 用于fa
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        # 对编码器输出进行层归一化
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        # zzy 0527 恢复原形
        if self.config.attention_type == "fa2":
            sequence_output = pad_input(sequence_output, indices, bs, seq_len)

        # 处理返回格式
        if not return_dict:
            return (sequence_output, ids_restore) + encoder_outputs[1:]
        
        return SiTMAEModelOutput(
            last_hidden_state=sequence_output,  # 最终隐藏状态
            mae_mask=mae_mask,                 # MAE掩码（用于重建任务）
            ids_restore=ids_restore,           # 恢复原始顺序的ID
            hidden_states=encoder_outputs.hidden_states,  # 各层隐藏状态（可选）
            attentions=encoder_outputs.attentions,        # 注意力权重（可选）
        )


class SiTMAEDecoder(nn.Module):
    """SiTMAE模型的解码器部分，负责将编码器的潜在表示重建为原始图像patch"""

    def __init__(self, config, num_patches):
        super().__init__()
        # 嵌入层（将编码器输出投影到解码器隐藏维度）
        self.decoder_embed = nn.Linear(
            config.hidden_size, config.decoder_hidden_size, bias=True
        )
        # 可学习的mask token（用于替换被mask的patch）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_position_embeddings = nn.Parameter(
            torch.randn(num_patches, config.decoder_hidden_size)
        )
        # 创建解码器专用的配置（调整部分参数）
        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        # 构建解码器层Transformer块（使用修改后的配置）
        self.decoder_layers = nn.ModuleList(
            [
                SiTMAELayer(decoder_config)
                for _ in range(config.decoder_num_hidden_layers)
            ]
        )
        # 层归一化
        self.decoder_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        # 最终预测层（将隐藏状态映射回原始patch空间）
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size[0]
            * config.patch_size[1]
            * config.num_channels,  # 输出维度=patch像素数×通道数
            bias=True,
        )
        # 训练相关设置
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        """初始化权重"""
        # 位置编码初始化
        for w in [self.decoder_position_embeddings]:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # mask token初始化（使用正态分布）
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,  # (batch, len_keep, hidden_size)
        patch_positions,  # patch的位置信息
        sample_ids_seq,
        ids_restore,  # 用于恢复原始patch顺序的索引 (bs, max_patches_len)
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 编码器最后的隐藏状态投影到解码器空间
        device = hidden_states.device
        x = self.decoder_embed(hidden_states)  # (bs, len_keep, decoder_hidden_size)
        bs, len_keep, decoder_hidden_size = x.shape
        max_patches_len = ids_restore.shape[1]
        # 将被掩码的patch替换为mask token
        mask_tokens = self.mask_token.repeat(bs, max_patches_len - len_keep, 1)
        x = torch.cat([x, mask_tokens], dim=1)  # 现在直接拼接、复原即可
        # 恢复原始patch顺序
        x = torch.gather(
            x,
            dim=1,
            index=ids_restore.unsqueeze(-1)
            .repeat(1, 1, decoder_hidden_size)
            .to(device),
        )
        # 添加位置编码
        hidden_states = x + self.decoder_position_embeddings[patch_positions]

        # zzy 0527 根据是否用fa生成不同变量
        if self.config.attention_type == "fa2":
            hidden_states, indices, cu_seqlens, max_seqlen = unpad_input(hidden_states, sample_ids_seq)
            attn_mask = None
        else:
            indices, cu_seqlens, max_seqlen = None, None, None
            attn_mask = SiTMAEEmbeddings.gen_attn_mask(sample_ids_seq, max_patches_len)

        # 通过解码器层
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                    mask=attn_mask,
                    # zzy 0520
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, 
                    head_mask=None, 
                    output_attentions=output_attentions,
                    mask=attn_mask,
                    # zzy 0520
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        # 最终层归一化
        hidden_states = self.decoder_norm(hidden_states)
        # 预测原始patch像素值
        logits = self.decoder_pred(hidden_states)

        # zzy 0527 恢复原形
        bs, seq_len, _ = x.shape
        if self.config.attention_type == "fa2":
            logits = pad_input(logits, indices, bs, seq_len)

        # 返回结果
        if not return_dict:
            return tuple(
                v
                for v in [logits, all_hidden_states, all_self_attentions]
                if v is not None
            )
        return SiTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SiTMAEForPreTraining(SiTMAEPreTrainedModel):
    """SiTMAE模型的预训练主类，包含完整的编码器-解码器结构和损失计算"""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sit = SiTMAEModel(config)  # SiTMAE主干
        self.decoder = SiTMAEDecoder(config, self.sit.embeddings.num_patches)  # 解码器
        # 初始化权重并应用最终处理
        self.post_init()

    def forward_loss(self, target, pred, mae_mask):
        """
        计算MAE预训练损失（掩码图像建模的像素重建损失）
        Args:
            target (torch.FloatTensor): 打包后的patch (batch_size, seq_len, patch_dim)
            pred (torch.FloatTensor): 预测的像素值 (batch_size, seq_len, patch_dim)
            mae_mask (torch.Tensor): 掩码标记 (batch_size, seq_len) 1表示masked
        Returns:
            torch.FloatTensor: 像素重建损失
        """
        # 可选：像素值归一化（稳定训练）
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        # 计算MSE损失
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # patch的平均损失 (bs, seq_len)
        # 只计算被mask部分的损失（通过mae_mask加权）
        loss = (loss * mae_mask).sum() / mae_mask.sum()
        return loss

    def forward(
        self,
        patches: torch.FloatTensor,  # 输入图像patch shape(bs, seq_len, patch_dim)
        patch_positions: torch.FloatTensor,  # patch位置信息
        sample_ids_seq: list[torch.Tensor],
        noise: Optional[torch.FloatTensor] = None,  # MAE噪声
        head_mask: Optional[torch.FloatTensor] = None,  # 注意力头掩码
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典格式
    ) -> Union[Tuple, SiTMAEForPreTrainingOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # 编码器前向传播
        outputs = self.sit(
            patches,
            patch_positions=patch_positions,
            noise=noise,
            sample_ids_seq=sample_ids_seq,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取编码器输出
        latent = outputs.last_hidden_state  # 潜在表示
        ids_restore = outputs.ids_restore  # 恢复原始顺序的索引
        mae_mask = outputs.mae_mask  # 掩码标记（仅用于损失计算）
        # 解码器前向传播
        decoder_outputs = self.decoder(
            latent,  # 编码器输出的潜在表示
            patch_positions,  # patch位置信息（一维）
            sample_ids_seq,
            ids_restore,  # 恢复原始顺序的索引
        )
        logits = decoder_outputs.logits  # 重建的patch像素值
        # 计算重建损失（仅在被mask的区域）
        loss = self.forward_loss(
            patches,
            logits,
            mae_mask,
        )
        # 处理返回结果
        if not return_dict:
            output = (logits, ids_restore) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SiTMAEForPreTrainingOutput(
            loss=loss,  # 重建损失
            logits=logits,  # 解码器输出
            ids_restore=ids_restore,  # 恢复索引
            hidden_states=outputs.hidden_states,  # 各层隐藏状态（可选）
            attentions=outputs.attentions,  # 注意力权重（可选）
        )


class SiTMAEModelWithoutMask(SiTMAEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = SiTMAEEmbeddings(config)
        self.encoder = SiTMAEEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        samples: torch.Tensor,
        patch_positions: torch.Tensor,
        sample_ids_seq: List[torch.Tensor],
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SiTMAEModelOutput]:
        # zzy 0520
        bs, seq_len, _ = samples.shape
        # 设置默认输出选项（这三个config都没定义）
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if samples is None:
            raise ValueError("You have to specify samples")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # zzy 0520 增加unmask_sample_ids_seq，即保留部分样本的长度标记
        patches, _, _, encoder_attn_mask, unmask_sample_ids_seq = self.embeddings(
            samples,
            patch_positions=patch_positions,   # 传入patch位置信息
            sample_ids_seq=sample_ids_seq,
            noise=noise,
            enable_mae_mask=False,
        )

        # zzy 0513 在嵌入之后将patch展开为没有padding的超长序列
        if self.config.attention_type == "fa2":
            patches, indices, cu_seqlens, max_seqlen = unpad_input(patches, unmask_sample_ids_seq)
        # zzy 0527
        else:
            cu_seqlens, max_seqlen = None, None

        encoder_outputs = self.encoder(
            patches,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask=encoder_attn_mask,
            # zzy 0513 用于fa
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )

        # Gabrielle 0528: 漏了一句
        sequence_output = encoder_outputs[0]
        # 恢复原形
        if self.config.attention_type == "fa2":
            sequence_output = self.layernorm(sequence_output)           # (bsz, num_patches, hidden_size)
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]
        return SiTMAEModelOutput(
            last_hidden_state=sequence_output,
            mae_mask=None,
            ids_restore=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SiTMAEForClassification(SiTMAEPreTrainedModel):
    def __init__(self, config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        # gwj 0506: 把 sit_nomask 改回 sit 使得可以正确读取权重
        # self.sit_nomask = SiTMAEModelWithoutMask(config)
        self.sit = SiTMAEModelWithoutMask(config)
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        patches: torch.FloatTensor,  # (b, 18, 16)
        patch_positions: torch.Tensor,
        sample_ids_seq: List[torch.Tensor],
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.sit(
            patches,
            patch_positions,
            sample_ids_seq,
            noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # (b, max_patches_len, hidden_size)

        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# hzh 0314: 增加多任务函数
class SiTForMultiTask(SiTMAEPreTrainedModel):
    def __init__(self, config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.sit = SiTMAEModelWithoutMask(config)
        self.classifier = (
            nn.Linear(config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )  # 这里假设
        self.regression_head_pulse_width = self._create_regression_head(
            config.hidden_size, out_nums=1
        )  # 脉冲宽度
        self.regression_head_pulse_time_delay = self._create_regression_head(
            config.hidden_size, out_nums=1
        )  # 脉冲时间延迟
        self.regression_head_pri = self._create_regression_head(
            config.hidden_size, out_nums=1
        )  # 脉冲重复间隔
        self.regression_head_num_pulses = self._create_regression_head(
            config.hidden_size, out_nums=1
        )  # 脉冲数
        self.regression_head_isr = self._create_regression_head(
            config.hidden_size, out_nums=1
        )  # 脉冲数

        # Initialize weights and apply final processing
        self.post_init()

    # hzh 0314: 创建一个通用回归头
    def _create_regression_head(self, hidden_size, out_nums):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            # nn.BatchNorm1d(1),  # channel = 1
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, out_nums),
        )

    def forward(
        self,
        patches: torch.FloatTensor,
        patch_positions: torch.Tensor,
        sample_ids_seq: List[torch.Tensor],
        noise: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.Tensor] = None,
        # pulse_width: Optional[torch.Tensor] = None,  # 脉冲宽度
        # pulse_time_delay: Optional[torch.Tensor] = None,  # 脉冲时间延迟
        # pri: Optional[torch.Tensor] = None,  # 脉冲重复间隔
        # num_pulses: Optional[torch.Tensor] = None,  # 脉冲数
        # isr: Optional[torch.Tensor] = None,  # gwj 0519: 新加回归目标
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.sit(
            patches,
            patch_positions,
            sample_ids_seq,
            noise,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # (b, 129, hidden_size) -> #(b, hidden_size)

        reg_logits = {}

        for key in labels.keys():
            if key == "classification":
                continue
            reg_head = f"regression_head_{key}"
            if hasattr(self, reg_head):
                regression_head = getattr(self, reg_head)
                # reg_head_logits = regression_head(
                #     sequence_output.unsqueeze(1)
                # )  # 扩维 (b, hidden_size) -> #(b, 1, hidden_size)
                reg_head_logits = regression_head(sequence_output)
                reg_logits[key] = reg_head_logits
                # reg_logits[key] = current_head_output.squeeze(-1)  # 存为(b,1)
            else:
                print(f"缺少回归头{reg_head}")

        logits = {}
        loss_details = {}
        label_weights = self.config.label_weights
        total_weight = sum(label_weights.values())
        label_weights = {
            key: value / total_weight for key, value in label_weights.items()
        }

        cls_logits = self.classifier(sequence_output)  # -> (b, num_labels)
        loss_fct = CrossEntropyLoss()
        cls_loss = loss_fct(cls_logits.view(-1, self.num_labels), labels['classification'].view(-1))
        loss_details["classification"] = cls_loss.item()
        loss = label_weights["classification"] * cls_loss

        # logits = cls_logits

        loss_fct = L1Loss()  # MAE loss
        for key in labels.keys():
            if key == 'classification':
                continue
            reg_per_loss = loss_fct(
                reg_logits[key].squeeze(), labels[key].squeeze()
            )
            loss += label_weights[key] * reg_per_loss
            loss_details[key] = reg_per_loss.item()

        # print("label_weights:", label_weights)

        logits.update({"classification": cls_logits})
        logits.update(reg_logits)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultiTaskOutput(
            loss=loss,
            logits=logits,  # 字典：分类和回归头的输出（动态）
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            loss_details=loss_details,  # 各任务的损失详情
        )
