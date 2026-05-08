# models/emmlm.py
from collections import defaultdict
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DSM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        residual = x
        x = self.fc(x)
        mask = self.gate(self.norm(x))
        return residual * (1 - mask) + x * mask

class EMMLLM(nn.Module):
    """
    支持两种训练数据形式：
        1. 前缀模式: <EM> 占位符在文本前面
        2. 插入模式: <IQ_START> ... <IQ_END> 包裹 EM token
    模型会自动将 projector 输出的 EM 特征替换这些占位 token。
    """

    def __init__(
        self,
        em_encoder: nn.Module,
        projector: nn.Module,
        language_model: nn.Module,
        pool_to_span: str = "avg",  # EM 特征长度与占位长度不一致时的处理方法
        freeze_em_encoder: bool = False,
        freeze_llm: bool = False
    ):
        super().__init__()

        self.em_encoder = em_encoder
        self.projector = projector
        self.lm = language_model
        self.pool_to_span = pool_to_span

        if freeze_em_encoder:
            for p in self.em_encoder.parameters():
                p.requires_grad = False
        if freeze_llm:
            for p in self.lm.model.parameters():
                p.requires_grad = False
            # for layer in self.lm.model.model.layers[-2:]:
            #     for p in layer.parameters():
            #         p.requires_grad = True

        tok = self.lm.tokenizer
        self.bos_id = tok.bos_token_id
        self.eos_id = tok.eos_token_id
        assert hasattr(self.lm, "hidden_size"), "language_model must expose .hidden_size"
        self.llm_hidden = self.lm.hidden_size

# def _adaptive_project_to_len(self, em_proj: torch.Tensor, tgt_len: int) -> torch.Tensor:
#     """
#     将 EM 特征序列适配到目标长度 tgt_len
#     """
#     if em_proj.dim() == 2:
#         em_proj = em_proj.unsqueeze(0)
#         squeeze_back = True
#     else:
#         squeeze_back = Falseke

#     B, P, H = em_proj.shape
#     if P == tgt_len:
#         out = em_proj
#     elif self.pool_to_span == "avg":
#         out = F.adaptive_avg_pool1d(em_proj.transpose(1, 2), tgt_len).transpose(1, 2)
#     else:
#         out = F.interpolate(em_proj.transpose(1, 2), size=tgt_len, mode="linear", align_corners=False).transpose(1, 2)

#     if squeeze_back:
#         out = out.squeeze(0)
#     return out

# Llava version by Dingwei
# def _merge_text_em(
#         self,
#         input_ids: torch.Tensor,
#         text_embeds: torch.Tensor,
#         em_proj: torch.Tensor
#     ) -> torch.Tensor:
#     """
#     将 EM 特征替换文本 embedding 中的占位 token
#     支持 <EM> 和 <IQ_START>/<IQ_END>
#     """
#     B, T, H = text_embeds.shape
#     device = text_embeds.device
#     mask_for_labels = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

#     new_text_embeds = text_embeds.clone()

#     for b in range(B):
#         ids = input_ids[b]

#         # 1. 替换 <EM>
#         if self.em_token_id is not None:
#             em_pos = (ids == self.em_token_id).nonzero(as_tuple=False).flatten()
#             if em_pos.numel() > 0:
#                 em_b = self._adaptive_project_to_len(em_proj[b], em_pos.numel())
#                 new_text_embeds[b, em_pos, :] = em_b
#                 mask_for_labels[b, em_pos] = True

#         # 2. 替换 <IQ_START> ... <IQ_END>
#         starts = (ids == self.iq_start_id).nonzero(as_tuple=False).flatten()
#         ends = (ids == self.iq_end_id).nonzero(as_tuple=False).flatten()
#         if starts.numel() > 0 and ends.numel() > 0:
#             s = int(starts[0].item())
#             e = int(ends[0].item())
#             if e - s - 1 > 0:
#                 em_seg = self._adaptive_project_to_len(em_proj[b], e - s - 1)
#                 new_text_embeds[b, s + 1:e, :] = em_seg
#                 mask_for_labels[b, s + 1:e] = True

#     return new_text_embeds, mask_for_labels

    def _merge_text_em(
        self,
        input_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        em_proj: torch.Tensor,
        patch_positions: torch.Tensor,
        patch_batch_idx: List[int]
    ):
        em_dict = defaultdict(list)
        for i, b_idx in enumerate(patch_batch_idx):
            mask = patch_positions[i] != 0  # (em_token_in_one_em,)
            valid_em = em_proj[i][mask]  # (num_valid, embed_dim)
            if valid_em.numel() > 0:
                em_dict[b_idx].append(valid_em)

        new_text_embeds = []
        for b in range(text_embeds.shape[0]):
            text_embed = text_embeds[b]  # (seq_len, embedding_size)
            ems = torch.cat(em_dict[b], dim=0).to(text_embed.dtype)  # (num_em_in_batch * em_token_in_one_em, embedding_size)

            # 找到这个 batch 中预留的 EM token 位置
            em_positions = (input_ids[b] == self.lm.em_token_id)

            assert len((input_ids[b] == self.lm.em_token_id).nonzero(as_tuple=True)[0]) == ems.size(0), "预留的 token 数量必须和 em embedding 数量一致"
            mask = em_positions.unsqueeze(-1)

            new_text_embeds.append(text_embed.masked_scatter(mask, ems))

        new_text_embeds = torch.stack(new_text_embeds, dim=0)
        return new_text_embeds

    def _mask_labels_for_positions(self, labels: torch.Tensor, mask_pos: torch.Tensor) -> torch.Tensor:
        if labels is None:
            return None
        labels = labels.clone()
        labels[mask_pos] = -100
        if self.bos_id is not None:
            labels[labels == self.bos_id] = -100
        return labels

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.lm.model, "gradient_checkpointing_enable"):
            self.lm.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )

    def gradient_checkpointing_disable(self):
        if hasattr(self.lm.model, "gradient_checkpointing_disable"):
            self.lm.model.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        samples=None,
        patch_batch_idx=None,
        patch_positions=None,
        sample_ids_seq=None,
        labels=None,
        return_intermediate=False,
    ):
        device = next(self.parameters()).device

        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        # 文本 embedding
        text_embeds = self.lm.model.get_input_embeddings()(input_ids)

        # EM 输入
        samples = samples.to(device)
        patch_positions = patch_positions.to(device)
        sample_ids_seq = sample_ids_seq

        em_feats = self.em_encoder(samples, patch_positions, sample_ids_seq)
        em_proj = self.projector(em_feats)
        # text_embeds, mask_for_labels = self._merge_text_em(input_ids, text_embeds, em_proj, patch_positions, batch["patch_batch_idx"])
        # labels = self._mask_labels_for_positions(labels, mask_for_labels)
        text_embeds = self._merge_text_em(input_ids, text_embeds, em_proj, patch_positions, patch_batch_idx)

        outputs = self.lm.model(inputs_embeds=text_embeds, attention_mask=attention_mask, labels=labels)

        if return_intermediate:
            # 返回完整的输出和中间值
            return {
                'outputs': outputs,
                'em_features': em_feats,
                'mlp_output': em_proj,
                'logits': outputs.logits,
                'loss': outputs.loss if hasattr(outputs, 'loss') else None
            }
        else:
            # 兼容原有接口
            return outputs


    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        samples: Optional[torch.Tensor] = None,
        patch_positions: Optional[torch.Tensor] = None,
        sample_ids_seq: Optional[torch.Tensor] = None,
        patch_batch_idx: Optional[List[int]] = None,
        max_new_tokens: int = 2048,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        **generate_kwargs
    ):
        """
        在带/不带 EM 的情况下做生成。
        - 如果提供 samples/patch_positions/... 则会计算 em_proj 并把它替换进文本 embedding（调用 _merge_text_em）。
        - 最终调用 self.lm.model.generate(..., inputs_embeds=...) 或者在没有 inputs_embeds 时传入 input_ids。
        返回 dict: {"sequences": Tensor, "texts": List[str]}
        """
        # 设备
        if device is None:
            device = next(self.parameters()).device

        tok = self.lm.tokenizer
        if bos_token_id is None:
            bos_token_id = getattr(self, "bos_id", tok.bos_token_id)
        if eos_token_id is None:
            eos_token_id = getattr(self, "eos_id", tok.eos_token_id)

        # 准备 input_ids / batch_size
        input_ids = input_ids.to(device)
        bsz = input_ids.shape[0]
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 取文本 embedding（可微分，但生成不需要梯度）
        embed_layer = self.lm.model.get_input_embeddings()
        text_embeds = embed_layer(input_ids).to(dtype=next(self.parameters()).dtype, device=device)

        # 如果有 EM 输入，则计算并合并
        if samples is not None and patch_positions is not None and sample_ids_seq is not None and patch_batch_idx is not None:
            samples = samples.to(device)
            patch_positions = patch_positions.to(device)
            # em_encoder 可能返回不同 shape，保持和 forward 一致

            em_feats = self.em_encoder(samples, patch_positions, sample_ids_seq)
            em_proj = self.projector(em_feats)
            # _merge_text_em 会检查 self.lm.em_token_id，且返回 new_text_embeds
            text_embeds = self._merge_text_em(input_ids, text_embeds, em_proj, patch_positions, patch_batch_idx)

        # 准备传给 generate 的参数
        gen_kwargs = dict(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
            pad_token_id=getattr(tok, "pad_token_id", None),
        )
        # 覆盖默认的任何额外参数
        gen_kwargs.update(generate_kwargs)

        # 调用底层模型的 generate（很多 HF 模型支持 inputs_embeds）
        # 注意：有些包装器可能是 self.lm.generate 或 self.lm.model.generate，优先尝试 model.generate
        model_for_generate = getattr(self.lm, "model", self.lm)
        sequences = model_for_generate.generate(**{k: v for k, v in gen_kwargs.items() if v is not None})

        # generated_ids = self._custom_generate(**{k: v for k, v in gen_kwargs.items() if v is not None})

        # sequences: (bsz, seq_len)
        # Decode
        texts = tok.batch_decode(sequences, skip_special_tokens=True)

        return texts

    def _custom_generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        eos_token_id: int,
        pad_token_id: int,
    ):
        """
        我们自己实现的 generate 函数，以替代 transformers 的版本。
        核心思想：
        1. 使用 KV 缓存 (past_key_values) 来避免重复计算，实现高效生成。
        2. 在每一步，模型只对最后一个 token 进行计算。
        3. 根据 do_sample 的值，选择贪心搜索或采样策略。
        """
        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        
        # 获取模型和词嵌入层，方便后续调用
        model_for_generate = getattr(self.lm, "model", self.lm)
        embed_layer = model_for_generate.get_input_embeddings()

        # 用于存储生成的 token ID
        generated_ids = torch.zeros(bsz, max_new_tokens, dtype=torch.long, device=device)
        
        # 步骤 1: 对输入的 prompt 进行一次前向传播，获取初始的 KV 缓存
        # `use_cache=True` 会让模型返回 past_key_values
        outputs = model_for_generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # 获取下一个 token 的 logits (在序列的最后一个位置)
        next_token_logits = logits[:, -1, :]

        # 标记每个序列是否已经生成结束
        is_finished = torch.zeros(bsz, 1, dtype=torch.bool, device=device)

        # 步骤 2: 自回归生成循环
        for i in range(max_new_tokens):
            # ---- 采样策略 ----
            if do_sample:
                # Temperature
                if temperature > 0 and temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Top-K
                if top_k > 0:
                    # 取 top_k 个 logits，其他的设置为 -inf
                    v, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')

                # Top-P (Nucleus Sampling)
                if top_p < 1.0:
                    probs = F.softmax(next_token_logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # 移除累计概率超过 top_p 的 token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 至少保留一个 token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove, -float('inf'))

                # 从修改后的 logits 分布中采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                # 贪心搜索 (Greedy Search)
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 如果一个序列已经结束，则后续生成的 token 用 pad_token_id 填充
            next_token_id = torch.where(is_finished, pad_token_id, next_token_id)
            
            # 记录生成的 token
            generated_ids[:, i] = next_token_id.squeeze(-1)
            
            # 检查是否生成了 EOS token
            is_finished = is_finished | (next_token_id == eos_token_id)
            
            # 如果所有序列都已结束，提前退出循环
            if is_finished.all():
                break

            # ---- 准备下一次迭代的输入 ----
            # 1. 获取新生成 token 的 embedding
            next_token_embeds = embed_layer(next_token_id)
            
            # 2. 将 attention_mask 扩展一位
            attention_mask = torch.cat(
                [attention_mask, torch.ones(bsz, 1, dtype=torch.long, device=device)], 
                dim=1
            )
            
            # 3. 进行下一次前向传播，并传入 KV 缓存
            #    注意：此时模型的输入 embeds 只有一个 token 的长度！
            outputs = model_for_generate(
                inputs_embeds=next_token_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token_logits = logits[:, -1, :]

        return generated_ids