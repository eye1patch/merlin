import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from utils import constants

class EMLanguageModel(nn.Module):
    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-8B",
                 lora_config: dict = {"use_lora": False}):
        super().__init__()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=None,
            trust_remote_code=True,
            # cache_dir="/home/shezhendong/tdw/EM-MLLM/qwen_checkpoints"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=False,
            trust_remote_code=True, 
            local_files_only=True,
            # cache_dir="/home/shezhendong/tdw/EM-MLLM/qwen_checkpoints"
        )

        if lora_config["use_lora"]:
            lora_config = LoraConfig(
                r=lora_config["r"],
                lora_alpha=lora_config["lora_alpha"],
                target_modules=lora_config["target_modules"],
                lora_dropout=lora_config["lora_dropout"],
                bias=lora_config["bias"],
                task_type=lora_config["task_type"]
            )
            self.model = get_peft_model(self.model, lora_config)
        
        # 添加特殊 token
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [constants.EM_TOKEN, constants.IQ_START_TOKEN, constants.IQ_END_TOKEN]
        })

        self.em_token_id = self.tokenizer.convert_tokens_to_ids(constants.EM_TOKEN)
        self.iq_start_id = self.tokenizer.convert_tokens_to_ids(constants.IQ_START_TOKEN)
        self.iq_end_id = self.tokenizer.convert_tokens_to_ids(constants.IQ_END_TOKEN)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))


        self.hidden_size = self.model.config.hidden_size

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels
        )
