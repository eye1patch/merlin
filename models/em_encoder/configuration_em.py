# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ViT MAE model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SiTMAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTMAEModel`]. It is used to instantiate an ViT
    MAE model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the ViT
    [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        decoder_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        decoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the decoder.
        decoder_num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the decoder.
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        mask_ratio (`float`, *optional*, defaults to 0.75):
            The ratio of the number of masked tokens in the input sequence.
        norm_pix_loss (`bool`, *optional*, defaults to `False`):
            Whether or not to train with normalized pixels (see Table 3 in the paper). Using normalized pixels improved
            representation quality in the experiments of the authors.

    Example:

    ```python
    >>> from transformers import ViTMAEConfig, ViTMAEModel

    >>> # Initializing a ViT MAE vit-mae-base style configuration
    >>> configuration = ViTMAEConfig()

    >>> # Initializing a model (with random weights) from the vit-mae-base style configuration
    >>> model = ViTMAEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "sit_mae"

    def __init__(
        self,
        hidden_size=768,                    # 每个patch被映射到768维的向量空间中
        num_hidden_layers=12,               # 12个Transformer块
        num_attention_heads=12,             # 编码器中注意力头的数量
        intermediate_size=3072,             # 编码器中FFN的维度
        hidden_act="gelu",                  # 编码器和池化层中的激活函数
        hidden_dropout_prob=0.0,            # 嵌入、编码器和池化层中所有全连接层的dropout概率
        attention_probs_dropout_prob=0.0,   # 注意力dropout概率
        initializer_range=0.02,             # 初始化所有权重矩阵的truncated_normal_initializer的标准差
        layer_norm_eps=1e-12,               # layer norm使用的epsilon
        max_seq_len = (4096,1),             # gwj: 需要传入最大图像尺寸,以计算pos_embed的宽度
        patch_size=(8,1),                   # patch size
        num_channels=2,                     # 通道数
        qkv_bias=True,                      # 是否对查询、键和值添加偏差
        decoder_num_attention_heads=16,     # 解码器中注意力头的数量
        decoder_hidden_size=512,            # 解码器输入维度
        decoder_num_hidden_layers=8,        # 8个Transformer块
        decoder_intermediate_size=2048,     # 解码器中FFN的维度
        mask_ratio=0.75,                    # 掩码率
        norm_pix_loss=False,                # 计算loss时是否对原图归一化
        group_images=False,
        use_fs=True,                        # 是否加入频率
        # zzy 0527
        attention_type="eager",             # Attention类型
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.max_seq_len = max_seq_len 
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.group_images = group_images
        self.use_fs = use_fs
        # zzy 0527
        self.attention_type = attention_type