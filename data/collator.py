# collator.py
from typing import List, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
import copy

from utils.utils import pack_sequence
from utils import constants

class EMCollator:
    """
    LLaVA风格的数据整理器：构造完整序列，只在response部分计算loss
    """
    def __init__(self, tokenizer, iq_max_seq_len, max_seq_len, patch_size):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.iq_max_seq_len = iq_max_seq_len
        self.patch_size = patch_size
        self.em_token = constants.EM_TOKEN
        self.iq_start_token = constants.IQ_START_TOKEN
        self.iq_end_token = constants.IQ_END_TOKEN
        self.text_iq_placeholder = constants.TEXT_IQ_PLACEHOLDER

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = [b for b in batch if b and isinstance(b, dict)]
        if len(batch) == 0:
            return None

        iq_data = []
        conversations = []

        for item in batch:
            if "samples" not in item or item["samples"] is None:
                continue

            # --------- 信号处理部分 ---------
            raw_signals = item["samples"]
            processed_signals_for_item = []
            num_patches_per_signal = [] # <-- 关键：存储当前item中每个信号的patch数量

            for single_raw_signal in raw_signals:
                sample = torch.tensor(single_raw_signal)
                sample_len = sample.shape[-2]

                # dropout 操作，确保长度可被 patch_size 整除
                patch_size = self.patch_size[0]
                num_drop = sample_len % patch_size
                if num_drop != 0:
                    # 为了可复现和稳定，建议使用截断而不是随机丢弃
                    # sample = sample[:, :-num_drop, :]
                    # 这里依然使用你的随机丢弃逻辑
                    drop_indices = torch.randn(sample_len).topk(num_drop).indices
                    mask = torch.ones(sample_len, dtype=torch.bool)
                    mask[drop_indices] = False
                    sample = sample[:, mask, :]

                # 计算这个信号将被切分成多少个patch(信号token)
                num_patches = sample.shape[-2] // patch_size
                num_patches_per_signal.append(num_patches)
                processed_signals_for_item.append(sample)
            
            # 将当前item的所有处理后信号作为一个整体，添加到batch数据中
            iq_data.append(processed_signals_for_item)

            # --------- 文本Prompt构建部分 ---------
            human_prompt = item["human_prompt"]
            split_parts = human_prompt.split(self.text_iq_placeholder)

            if len(split_parts) == 1 and len(num_patches_per_signal) == 1:
                num_patches = num_patches_per_signal[0] - 1
                em_tokens = "".join([self.em_token] * num_patches)
                new_human_prompt_parts = [f"{self.iq_start_token}{em_tokens}{self.iq_end_token}"]
                new_human_prompt_parts.append(split_parts[0])
            elif len(split_parts) == 2 and len(num_patches_per_signal) == 1:
                num_patches = num_patches_per_signal[0] - 1
                em_tokens = "".join([self.em_token] * num_patches)
                new_human_prompt_parts = [f"{self.iq_start_token}{em_tokens}{self.iq_end_token}"]
                new_human_prompt_parts.extend(split_parts[0])
            else:
                 # 断言检查，确保 <iq_data> 占位符数量和实际信号数量匹配
                assert len(split_parts) - 1 == len(raw_signals), \
                    f"Mismatch: {len(split_parts)-1} '{self.text_iq_placeholder}' placeholders vs {len(raw_signals)} samples provided."
                new_human_prompt_parts = ["Human: "]
                for i, part in enumerate(split_parts):
                    new_human_prompt_parts.append(part)
                    if i < len(num_patches_per_signal):
                        # 使用刚才计算出的正确patch数量来生成占位符, -1是去掉cls token的位置
                        num_patches = num_patches_per_signal[i] - 1
                        em_tokens = "".join([self.em_token] * num_patches)
                        
                        # 拼接上IQ信号的特殊token边界和占位符
                        new_human_prompt_parts.append(
                            f"{self.iq_start_token}{em_tokens}{self.iq_end_token}"
                        )
            
            new_human_prompt = "".join(new_human_prompt_parts)

            conversations.append([{"from": "human", "value": new_human_prompt}, {"from": "gpt", "value": item['gpt_response']}])

            # 构造完整对话和单独的prompt
            # full_conversation = f"{new_human_prompt}\nAssistant: {item['gpt_response']}"
            # prompt_part = f"{new_human_prompt}\nAssistant: "

            # conversations.append(full_conversation)
            # prompts_only.append(prompt_part)

        # Gabrielle 0516: 打包之前预留cls和fs,增大max_seq_len
        iq_max_seq_len = self.iq_max_seq_len[0] + self.patch_size[0]

        flat_samples, parent_ids = [], []
        for b_idx, seq in enumerate(iq_data):
            for sig in seq:
                flat_samples.append([sig]) # 适配pack_sequence
                parent_ids.append(b_idx)
        
        patches, patch_positions, sample_ids_seq = pack_sequence(
            flat_samples,    #列表 len(iq_data) = bs
            self.patch_size[0],
            iq_max_seq_len,
            enable_packing=False,
        )

        # # 3. Token化完整对话
        # encoding = self.tokenizer(
        #     conversations,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=self.max_seq_len,
        #     add_special_tokens=True
        # )

        # input_ids = encoding["input_ids"]
        # attention_mask = encoding["attention_mask"]

        # # 4. 构造labels：和input_ids等长，但只在response部分计算loss
        # labels = input_ids.clone()

        # # 对prompt部分进行tokenize以获得其确切长度
        # prompt_encodings = self.tokenizer(
        #     prompts_only,
        #     return_tensors="pt",
        #     padding="longest",
        #     truncation=True,
        #     max_length=self.max_seq_len,
        #     add_special_tokens=True
        # )
        # prompt_lengths = prompt_encodings.attention_mask.sum(dim=1)

        # for i, length in enumerate(prompt_lengths):
        #     # 将每个样本的prompt部分（包括开头的special token和结尾的Assistant:）设置为-100
        #     labels[i, :length] = -100

        # # Mask掉padding部分
        # labels[attention_mask == 0] = -100

        input_ids, attention_mask, labels = self.templateize_and_tokenize(conversations)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "samples": patches,
            "patch_batch_idx": parent_ids,
            "patch_positions": patch_positions,
            "sample_ids_seq": sample_ids_seq,
            "labels": labels
        }
    
    def templateize_and_tokenize(self, conversations):
        input_ids, attention_mask, labels = preprocess_qwen3(conversations, self.tokenizer, self.max_seq_len, infer=False)
        return input_ids, attention_mask, labels
            
class EMInferCollator(EMCollator):
    def __init__(self, tokenizer, iq_max_seq_len, max_seq_len, patch_size):
        super().__init__(tokenizer, iq_max_seq_len, max_seq_len, patch_size)
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        collator_result = super().__call__(batch)
        collator_result["id"], collator_result["human_prompt"], collator_result["answer"], collator_result["snr"] = [], [], [], []
        for item in batch:
            collator_result["id"].append(item["id"])
            collator_result["human_prompt"].append(item["human_prompt"])
            collator_result["answer"].append(item["answer"])
            collator_result["snr"].append(item["snr"])
        return collator_result
    
    def templateize_and_tokenize(self, conversations):
        input_ids, attention_mask, labels = preprocess_qwen3(conversations, self.tokenizer, self.max_seq_len, infer=True)
        return input_ids, attention_mask, labels

def preprocess_qwen3(sources, tokenizer, max_len=2048, infer=False, padding_left=True):
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token

    em_token_index = tokenizer.convert_tokens_to_ids(constants.EM_TOKEN)
    iq_start_token_index = tokenizer.convert_tokens_to_ids(constants.IQ_START_TOKEN)
    iq_end_token_index = tokenizer.convert_tokens_to_ids(constants.IQ_END_TOKEN)
    # im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_token = tokenizer.convert_tokens_to_ids("\n")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_start = tokenizer.convert_tokens_to_ids("<think>")
    think_end = tokenizer.convert_tokens_to_ids("</think>")
    # unmask_tokens_idx =  [
    #     nl_token, im_start, im_end
    #     ]

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    #### 修改部分 ####
    pretrain_template = """
    {% for m in messages %}
        {% if m['role'] != 'user' and m['content'] | length > 0 %}
            {{ '<|im_start|>' + m['content'].strip() + '<|im_end|>' + '\n' }}
        {% endif %}
    {% endfor %}
    """
    tokenizer.chat_template = chat_template
    use_pretrain_template = False
    # Apply prompt templates
    input_ids, targets, attention_mask = [], [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]
            role =  roles.get(role, role)
            conv = [{"role" : role, "content" : content}]
            #### 根据user是否是空更换模板 ####
            if role == 'user':
                if len(content) == 0:
                    #### pretrain data ####
                    tokenizer.chat_template = pretrain_template
                    use_pretrain_template = True
                else:
                    tokenizer.chat_template = chat_template
                    use_pretrain_template = False

            if use_pretrain_template:
                ret = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                encode_id = tokenizer.apply_chat_template(conv, add_generation_prompt=False, enable_thinking=False)
                input_id += encode_id
                if role in ["user", "system"]:
                    target += [constants.IGNORE_INDEX] * len(encode_id)
                else:
                    target += encode_id
            else:
                tokenizer.chat_template = chat_template
                ret = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                encode_id = tokenizer.apply_chat_template(conv, add_generation_prompt=False, enable_thinking=False)
                
                if role in ["user", "system"]:
                    target += [constants.IGNORE_INDEX] * len(encode_id)
                    input_id += encode_id
                elif role in ["assistant"]:
                    think_part_id = tokenizer.encode('<think>\n\n</think>\n\n')
                    encode_id = encode_id[:3] + think_part_id + encode_id[3:]
                    if infer:
                        encode_id = encode_id[:-2]
                    target += [constants.IGNORE_INDEX] * 7 + encode_id[7:]
                    input_id += encode_id
                else:
                    import pdb; pdb.set_trace()

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        # for idx, encode_id in enumerate(input_id):
        #     if encode_id in unmask_tokens_idx:
        #         target[idx] = encode_id
        # assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        
        input_id = input_id[:max_len]
        target = target[:max_len]
            
        input_ids.append(input_id)
        targets.append(target)
    
    max_batch_len = max(len(ids) for ids in input_ids)
    attention_masks, padded_input_ids, padded_targets = [], [], []

    for input_id, target in zip(input_ids, targets):
        # 计算需要 padding 的长度
        padding_len = max_batch_len - len(input_id)
        
        # 创建 attention mask
        # 真实 token 的位置是 1，padding 的位置是 0
        if padding_left:
            attn_mask = [0] * padding_len + [1] * len(input_id)
            padded_input_id = [tokenizer.pad_token_id] * padding_len + input_id
            padded_target = [constants.IGNORE_INDEX] * padding_len + target
        else:
            attn_mask = [1] * len(input_id) + [0] * padding_len
            padded_input_id = input_id + [tokenizer.pad_token_id] * padding_len
            padded_target = target + [constants.IGNORE_INDEX] * padding_len
        attention_masks.append(attn_mask)
        
        # 对 input_ids 进行 padding
        # 使用 tokenizer.pad_token_id
        padded_input_ids.append(padded_input_id)
        
        # 对 targets/labels 进行 padding
        # 通常使用 IGNORE_INDEX 进行 padding
        padded_targets.append(padded_target)
    #for OOM https://github.com/LLaVA-VL/LLaVA-NeXT/issues/352
    del tokenizer
    return torch.tensor(padded_input_ids, dtype=torch.long), torch.tensor(attention_masks, dtype=torch.long), torch.tensor(padded_targets, dtype=torch.long)
