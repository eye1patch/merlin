from transformers.activations import ACT2FN
import torch.nn as nn
import torch.nn.functional as F

class EMProjector(nn.Module):
    def __init__(self, em_hidden_size: int, llm_hidden_size: int, projector_dims: int,
                 proj_seq_len: int = None, activation: str = "gelu"):
        super().__init__()
        self.proj_seq_len = proj_seq_len
        self.act_fn = ACT2FN[activation]  

        layers = []
        in_dim = em_hidden_size
        # print(in_dim)
        # for i in range(num_layers - 1):
        #     layers.append(nn.Linear(in_dim, llm_hidden_size))
        #     layers.append(nn.ReLU() if self.act_fn is None else nn.Identity())  
        #     in_dim = llm_hidden_size
        layers.append(nn.Linear(in_dim, llm_hidden_size))
        self.layers = nn.ModuleList([
            nn.Linear(em_hidden_size, projector_dims),
            nn.ReLU() if self.act_fn is None else nn.Identity(),
            nn.Linear(projector_dims, llm_hidden_size)
        ])

        self.norm = nn.LayerNorm(llm_hidden_size)

        # self.proj_seq_len = proj_seq_len
        # self.act_fn = ACT2FN[activation]  

        # layers = []
        # in_dim = em_hidden_size
        # # print(in_dim)
        # for i in range(1 - 1):
        #     layers.append(nn.Linear(in_dim, llm_hidden_size))
        #     layers.append(nn.ReLU() if self.act_fn is None else nn.Identity())  
        #     in_dim = llm_hidden_size
        # layers.append(nn.Linear(in_dim, llm_hidden_size))
        # self.layers = nn.ModuleList(layers)

        # self.norm = nn.LayerNorm(llm_hidden_size)

    def forward(self, em_features):
        x = em_features
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.act_fn(x) if layer is not self.layers[-1] else x  
        x = self.norm(x)

        # if self.proj_seq_len is not None and self.proj_seq_len != x.size(1):
        #     x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.proj_seq_len).transpose(1, 2)

        return x
