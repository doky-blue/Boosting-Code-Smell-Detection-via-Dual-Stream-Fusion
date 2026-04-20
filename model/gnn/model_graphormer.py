# file: model_graphormer.py

import torch.nn as nn
from graphormer.models.graphormer import GraphormerGraphEncoder
import torch


class PretrainedNodeFeature(nn.Module):
    """
    (This module is from the previous step and is correct)
    An auxiliary module to replace self.encoder.graph_node_feature.

    ✅ **已改回**:
    它现在从 'x' 键读取数据。
    """

    def __init__(self):
        super().__init__()

    def forward(self, batched_data):
        # ✅ **修改点**:
        # 改回从 'x' 键读取。
        # 在 GraphormerClassifier.forward 中, 'x' 键将被
        # 替换为 3D 浮点数张量。
        return batched_data['x']

class GraphormerClassifier(nn.Module):
    def __init__(self,
                 num_classes: int,
                 vocab_size: int,
                 embed_dim: int = 512,
                 ffn_dim: int = 1024,
                 num_layers: int = 16,
                 num_heads: int = 16,
                 dropout: float = 0.1,
                 use_pretrained: bool = False):  # ✅ 步骤 1: 添加新的初始化参数
        super().__init__()
        self.use_pretrained = use_pretrained

        self.encoder = GraphormerGraphEncoder(
            num_atoms=vocab_size,
            num_in_degree=512,
            num_out_degree=512,
            num_edges=5,
            num_spatial=512,
            num_edge_dis=128,
            edge_type="bond",
            multi_hop_max_dist=5,
            num_encoder_layers=num_layers,
            embedding_dim=embed_dim,
            ffn_embedding_dim=ffn_dim,
            num_attention_heads=num_heads,
            dropout=dropout,
        )

        if self.use_pretrained:
            self.encoder.graph_node_feature = PretrainedNodeFeature()

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, batched_data):

        # ✅ **修改点**:
        # 我们必须创建一个副本并替换 'x' 键
        forward_data = batched_data.copy()

        if self.use_pretrained:
            if 'node_feature' not in forward_data:
                raise ValueError("Model is in pre-trained mode, but 'node_feature' not found in batch.")

            # ✅ **核心修复**:
            # 将 3D 浮点数张量 [B, T, D] 放入 'x' 键。
            # 这是为了满足 graphormer_graph_encoder.py:204 的
            # (data_x[:, :, 0]).eq(0)
            forward_data['x'] = forward_data['node_feature']

        # Pass the *modified* dictionary to the encoder.
        # PretrainedNodeFeature 模块也将读取 forward_data['x']
        output = self.encoder(forward_data)

        if isinstance(output, (list, tuple)):
            graph_rep = output[-1]
        else:
            graph_rep = output

        return self.classifier(graph_rep)