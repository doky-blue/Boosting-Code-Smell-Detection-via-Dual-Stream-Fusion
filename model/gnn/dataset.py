import json
from torch.utils.data import Dataset
import torch
import os
import linecache
from torch_geometric.data import Data
import networkx as nx

# 请确保 utils.py 中的 NODE_TYPE_VOCAB 已经加载
# from utils import NODE_TYPE_VOCAB, UNK
NODE_TYPE_VOCAB = json.load(open("spoon_vocab.json"))
UNK = NODE_TYPE_VOCAB.get("<UNK>", 0)


class JavaGraphDataset(Dataset):
    num_classes = 1

    def __init__(self, jsonl_path, max_nodes=512, graph_base_dir="output_graphs/new",
                 pretrained_embed_path=None):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.max_nodes = max_nodes
        self.graph_base_dir = graph_base_dir
        with open(jsonl_path) as f:
            self.n = sum(1 for _ in f)

        # self.split_name = os.path.basename(jsonl_path).split('.')[0]
        self.type = os.path.basename(self.jsonl_path).split('.')[0].split('_')[1]

        # ✅ 新增：加载预训练词嵌入
        if pretrained_embed_path:
            self.pretrained_embeddings = torch.load(pretrained_embed_path)
            self.vocab_to_idx = {v: k for k, v in NODE_TYPE_VOCAB.items()}
            # 检查嵌入维度是否与模型一致
            self.embed_dim = self.pretrained_embeddings.shape[1]
        else:
            self.pretrained_embeddings = None
            self.embed_dim = 1  # 原始的整数ID模式

    def __len__(self):
        return self.n

    def get(self, idx):
        line = ""
        while not line.strip():
            line = linecache.getline(self.jsonl_path, idx + 1)
            if not line.strip():
                idx += 1
                if idx + 1 > self.n:
                    raise StopIteration("End of file reached.")

        item = json.loads(line)

        graph_file_path = os.path.join(self.graph_base_dir, self.type, f"snippet_{idx}_super_graph.json")
        if not os.path.exists(graph_file_path):
            raise FileNotFoundError(f"Graph file not found: {graph_file_path}")

        with open(graph_file_path, "r") as f:
            super_graph = json.load(f)

        graph_data = self.super_graph_to_graphormer(super_graph, self.max_nodes)

        # ... (验证 num_nodes 和添加 max_nodes 的代码保持不变) ...
        if 'num_nodes' in graph_data:
            num_nodes_tensor = graph_data['num_nodes']
            num_nodes = num_nodes_tensor.item() if isinstance(num_nodes_tensor, torch.Tensor) else num_nodes_tensor
        else:
            num_nodes = len(super_graph['nodes'])
        num_nodes = min(num_nodes, self.max_nodes)
        graph_data['max_nodes'] = torch.tensor([self.max_nodes])
        graph_data['num_nodes'] = torch.tensor([num_nodes])

        data_obj = Data()
        for key, value in graph_data.items():
            setattr(data_obj, key, value)

        # 假设二分类任务的标签在 item["label"] 中存储为 0 或 1
        labels = item["label"]

        if isinstance(labels, list):
            # 如果标签是 [0] 或 [1]
            y_val = float(labels[0])
        else:
            # 如果标签是 0 或 1
            y_val = float(labels)

        # 确保 y 是一个形状为 [1] 的张量，以便 collate_fn 堆叠为 [B, 1]
        data_obj.y = torch.tensor([y_val], dtype=torch.float)

        return data_obj

    def __getitem__(self, idx):
        return self.get(idx)

        # file: dataset.py

    def super_graph_to_graphormer(self, graph, max_nodes):
        node_map = {node['id']: i for i, node in enumerate(graph['nodes'])}
        num_nodes = len(graph['nodes'])

        # 步骤 1: 始终创建节点类型ID的张量 (LongTensor)
        x_ids = torch.zeros(max_nodes, dtype=torch.long)
        for i, node in enumerate(graph['nodes']):
            if i < max_nodes:
                x_ids[i] = NODE_TYPE_VOCAB.get(node['type'], UNK)

            # 步骤 2: 如果有预训练嵌入，则创建节点特征张量 (FloatTensor)
        node_feature = None
        if self.pretrained_embeddings is not None:
            node_feature = torch.zeros(max_nodes, self.embed_dim, dtype=torch.float)
            for i, node in enumerate(graph['nodes']):
                if i < max_nodes:
                    # 使用已经创建的 x_ids 来安全地索引嵌入
                    vocab_idx = x_ids[i].item()
                    node_feature[i] = self.pretrained_embeddings[vocab_idx]

        attn_bias = torch.zeros(max_nodes + 1, max_nodes + 1)
        edge_type_map = {"AST_PARENT_CHILD": 1, "CFG_NEXT": 2, "DFG_DEF_USE": 3}
        attn_edge_type = torch.zeros(max_nodes, max_nodes, dtype=torch.long)

        for edge in graph['edges']:
            src_id = node_map.get(edge['source'])
            dst_id = node_map.get(edge['target'])
            if src_id is not None and dst_id is not None and src_id < max_nodes and dst_id < max_nodes:
                edge_type = edge_type_map.get(edge['type'], 0)
                attn_edge_type[src_id, dst_id] = edge_type
                attn_bias[src_id, dst_id] = edge_type

        in_degree = torch.zeros(max_nodes, dtype=torch.long)
        out_degree = torch.zeros(max_nodes, dtype=torch.long)
        for src, dst in zip(*attn_edge_type.nonzero(as_tuple=True)):
            in_degree[dst] += 1
            out_degree[src] += 1

        max_degree = 511
        in_degree = torch.clamp(in_degree, max=max_degree)
        out_degree = torch.clamp(out_degree, max=max_degree)

        spatial_pos = self._get_spatial_pos(graph, max_nodes)
        max_spatial_dist = 511
        spatial_pos = torch.clamp(spatial_pos, max=max_spatial_dist)

            # 步骤 3: 构造返回字典
        data = {
            'x': x_ids,  # 'x' 现在是节点类型ID
            'attn_bias': attn_bias,
            'attn_edge_type': attn_edge_type.long(),
            'in_degree': in_degree,
            'out_degree': out_degree,
            'spatial_pos': spatial_pos,
            'edge_input': torch.zeros(max_nodes, max_nodes, 5, 1, dtype=torch.long),
            'num_nodes': torch.tensor([num_nodes])
        }

        # 如果存在节点特征，则加入字典
        if node_feature is not None:
            data['node_feature'] = node_feature
        return data



    # 新增函数：计算最短路径距离
    def _get_spatial_pos(self, graph, max_nodes):
        node_map = {node['id']: i for i, node in enumerate(graph['nodes'])}
        num_nodes = len(graph['nodes'])

        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for edge in graph['edges']:
            src_id = node_map.get(edge['source'])
            dst_id = node_map.get(edge['target'])
            if src_id is not None and dst_id is not None and src_id < num_nodes and dst_id < num_nodes:
                G.add_edge(src_id, dst_id)

        # 将无法到达的节点距离设置为 max_nodes（或任何大值）
        spatial_pos = torch.full((num_nodes, num_nodes), max_nodes, dtype=torch.long)

        for i in range(num_nodes):
            try:
                # 计算从节点 i 到所有其他节点的最短路径
                path_lengths = nx.shortest_path_length(G, source=i)
                for j in range(num_nodes):
                    length = path_lengths.get(j, max_nodes)
                    spatial_pos[i, j] = length
            except nx.NetworkXNoPath:
                # 如果没有路径，距离设置为最大值
                spatial_pos[i, :] = max_nodes

        # Pad the tensor to max_nodes x max_nodes
        padded_spatial_pos = torch.zeros(max_nodes, max_nodes, dtype=torch.long)
        padded_spatial_pos[:num_nodes, :num_nodes] = spatial_pos

        return padded_spatial_pos