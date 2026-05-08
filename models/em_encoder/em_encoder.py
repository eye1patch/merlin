import pathlib
import logging
from typing import List

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .modeling_em import SiTMAEModelWithoutMask, SiTMAEConfig, SiTMAEModel

logger = logging.getLogger(__name__)

def init_sit(model_args):
    """
    初始化电磁信号核心encoder
    """
    # Load or create config
    if model_args.model_name_or_path:
        config = SiTMAEConfig.from_pretrained(
            model_args.model_name_or_path
        )
    else:
        config = SiTMAEConfig()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    config.update(
        {
            "max_seq_len": (model_args.max_seq_len if model_args.max_seq_len is not None else [4096, 1]),
            "patch_size": model_args.patch_size,
            "use_fs": model_args.use_fs,  # Gabrielle 0520: 增加 use_fs 传参
            "attention_type": model_args.attention_type, # Gabrielle 0527: 增加 attention_type 传参
        }
    )

    if model_args.model_name_or_path:
        model = load_sit_model(config, model_args.model_name_or_path)
    else:
        model = SiTMAEModelWithoutMask(config)
        logger.info("Training new model from scratch")

    return model

def load_sit_model(config, model_path):
    
    model = SiTMAEModelWithoutMask(config)
    pretrained_weights = load_file(
        pathlib.Path(model_path).joinpath("model.safetensors")
    )
    model_dict = model.state_dict()
    for key in model_dict.keys():
        pretrained_key = "sit." + key
        if pretrained_key in pretrained_weights:

            pretrained_value = pretrained_weights[pretrained_key]

            if key == "embeddings.position_embeddings":
                target_shape = model_dict[key].shape
                if pretrained_value.shape != target_shape:
                    logger.warning(
                        f"Adapting position embeddings from {pretrained_value.shape} to {target_shape}"
                    )
                pretrained_value = pretrained_value[: target_shape[0], :]

            model_dict[key] = pretrained_value
    model.load_state_dict(model_dict, strict=True)

    return model

class EMEncoder(nn.Module):
    """
    电磁信号编码包装类
    """
    def __init__(self, sit_model: SiTMAEModelWithoutMask, **encoder_wkargs):
        """
        初始化函数
        :param sit_model: 实例化的SiT模型
        :param encoder_kwargs: encoder所需参数
        """
        super().__init__()
        self.sit = sit_model
    
    def forward(self, samples: torch.Tensor, patch_positions: torch.Tensor, sample_ids_seq: List[torch.Tensor]):
        """
        前向传播
        :param samples: 电磁信号输入, shape: (B, C, L)
        :param patch_positions: 电磁信号patch位置, shape: (B, num_patches)
        :param sample_ids_seq: 标记每个样本真实长度的列表, shape: (B, L)
        :return: 电磁信号特征表示, shape: (B, seq_len, hidden_size)
        """
        encoder_outputs = self.sit(
            samples=samples,
            patch_positions=patch_positions,
            sample_ids_seq=sample_ids_seq,
            return_dict=True
        )
        sequence_output = encoder_outputs.last_hidden_state
        
        features = self.sit.layernorm(sequence_output)
        return features

if __name__ == "__main__":
    class TestArgs:
        model_name_or_path = "/path/to/MERLIN/encoder_checkpoint"
        config_overrides = None
        max_seq_len = (4096, 1)
        patch_size = (8, 1)
        use_fs = False
        attention_type = "eager"

    sit = init_sit(TestArgs())
    em_encoder = EMEncoder(sit)
    em_encoder.eval()

    batch_size, seq_len, patch_dim = 16, 4096 // 8 + 1, 8 * 1 * 2 # 8个采样点一个token，seq_len: 最大4096个点 // 8 + 1个[cls]，patch_dim: 8个采样点 * 1 * 2个通道


    samples = torch.randn(batch_size, seq_len, patch_dim)
    patch_positions = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1)

    sample_ids_seq = [torch.tensor([seq_len]) for _ in range(batch_size)]
    with torch.no_grad():
        features = em_encoder(
            samples, patch_positions, sample_ids_seq
        )
    print(features.shape)