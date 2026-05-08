# dataset.py
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import random
from tqdm import tqdm
import logging

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

# prepare_dataset.py
import json
from datasets import Dataset, Features, Value, Sequence
from pathlib import Path
from tqdm import tqdm

from utils.utils import transform_downstream

def build_hf_dataset(origin_data_path, df_dataset_path):

    def data_generator(pbar):
        """一个生成器，逐条读取并解析jsonl文件"""
        for file in origin_data_path.glob("*.jsonl"):
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line.strip())
                        first_element = data['iq_data'][0]
                        if isinstance(first_element, list):
                            second_level_element = first_element[0]
                            if not isinstance(second_level_element, list):
                                # 这是一个二维数组, 将它包装成三维
                                data['iq_data'] = [data['iq_data']]
                        
                        yield data
                        pbar.update(1)

    # 定义数据集的结构，这对于处理复杂类型（如嵌套的对话）很有帮助
    # 注意：IQ数据我们先作为列表加载，后面再转为Tensor
    features = Features({
        'id': Value('string'),
        'conversations': [{'from': Value('string'), 'value': Value('string')}],
        'iq_data': Sequence(Sequence(Sequence(Value('float32'))))
    })

    # 从生成器创建Dataset对象
    # 这会处理所有数据并将其写入磁盘上的Arrow文件中
    # 对于非常大的数据集，这一步可能会花一些时间，但只用做一次
    with tqdm() as pbar:
        raw_dataset = Dataset.from_generator(data_generator, features=features)

    def preprocess_function(sample):
        """
        这个函数替代了您原来 EMTextDataset 的大部分逻辑。
        它负责解析对话，并将IQ数据转换为Tensor。
        """
        # 1. 解析对话
        human_prompt = ""
        gpt_response = ""
        for conv in sample["conversations"]:
            if conv["from"] == "human":
                human_prompt = conv["value"]
            elif conv["from"] == "gpt":
                gpt_response = conv["value"]
        
        # 返回新的字段
        return {
            "id": sample["id"],
            "human_prompt": human_prompt,
            "gpt_response": gpt_response,
            # IQ数据在这里还保持为列表，collator中再转为Tensor更高效
            "samples": sample["iq_data"]
        }

    # 使用 .map() 方法应用预处理函数
    # num_proc > 1 可以开启多进程并行处理，极大加快速度
    processed_dataset = raw_dataset.map(
        preprocess_function,
        num_proc=32, # 使用8个CPU核心来加速
        remove_columns=raw_dataset.column_names # 删掉不再需要的原始列
    )

    # train_dataset = train_dataset.select(range(10000))

    config = {
        "em_encoder": {
            "patch_size": [8, 1]
        },
        "iq_col_name": "samples",
        "batch_size": 4,
        "num_proc": 32
    }

    train_dataset = processed_dataset.map(
        lambda x: transform_downstream(
            x,
            config["em_encoder"]["patch_size"],
            config["iq_col_name"]
        ),
        batched=True,
        batch_size=config["batch_size"],
        num_proc=config["num_proc"]
    )

    # 将处理好的数据集保存到磁盘
    # 这会生成一个包含Arrow文件和元数据的文件夹
    train_dataset.save_to_disk(df_dataset_path)

    print(f"Dataset saved to {df_dataset_path}")

logger = logging.getLogger(__name__)

class EMTextDataset(Dataset):
    """
    每条样本包含：
      - conversations: 对话列表
      - iq_data: 电磁信号列表（每个样本的 IQ 数据）
    针对projector训练，构造input(human prompt + IQ) -> output(gpt response)的标签
    """
    def __init__(self, dataset_path: str, tokenizer, mix_strategy: str, max_seq_len: int = 512):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mix_strategy = mix_strategy

        # 根据策略加载和混合样本
        self.samples = []
        self._load_and_mix_samples()

    def _load_and_mix_samples(self):
        
        if self.mix_strategy not in ["sequential", "shuffle"]:
            raise ValueError(f"Unknown mixing strategy: {self.mix_strategy}")
        

        logger.info(f"Loading samples from: {str(self.dataset_path)}")
        for file in tqdm(self.dataset_path.glob("*.jsonl")):
            with open(file, "r", encoding="utf-8") as f:
                for line in tqdm(f):
                    if line.strip():
                        self.samples.append(json.loads(line.strip()))
        if self.mix_strategy == "shuffle":
            logger.info("Shuffling all loaded samples...")
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _extract_conversation_parts(self, conversations: List[Dict[str, str]]) -> tuple:
        """
        提取对话中的human问题和gpt回答
        返回: (human_prompt, gpt_response)
        """
        human_prompt = ""
        gpt_response = ""
        
        for conv in conversations:
            role = conv["from"]
            value = conv["value"]
            if role == "human":
                human_prompt = value
            elif role == "gpt":
                gpt_response = value
                
        return human_prompt, gpt_response

    def __getitem__(self, idx):
        sample = self.samples[idx]

        assert "conversations" in sample, f"Bad sample at idx {idx}: {sample}"


        # 提取对话内容
        conversations = sample["conversations"]
        human_prompt, gpt_response = self._extract_conversation_parts(conversations)

        # EM signal
        iq_data = sample.get("iq_data", [])
        iq_tensor = torch.tensor(iq_data, dtype=torch.float)  # shape: [L, 2]

        result = {
            "id": sample.get("id", str(idx)),
            "human_prompt": human_prompt,      # 输入：人类问题
            "gpt_response": gpt_response,      # 输出：GPT回答（作为label）
            "samples": iq_tensor               # 输入：EM 信号
        }

        assert result["human_prompt"] != "" and result["gpt_response"] != "", f"Empty prompt/response at idx {idx}"
        return result



