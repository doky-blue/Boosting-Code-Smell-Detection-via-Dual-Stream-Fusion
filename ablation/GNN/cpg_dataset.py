import os
import re
import torch
import networkx as nx
import numpy as np
from glob import glob
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

# =================================================================
# 1. 配置和编码映射
# =================================================================

# 1.1 定义节点类型映射 (Joern CPG 中常见的节点类型)
# 将所有可能出现的 Joern CPG 节点标签映射到唯一的整数 ID。
node_labels_path = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/node_labels.txt"
NODE_LABELS = ['UNKNOWN_NODE']
with open(node_labels_path,'r') as f:
    for line in f.readlines():
        NODE_LABELS.append(line.strip())
# NODE_LABELS = [
#     'META_DATA', 'FILE', 'NAMESPACE_BLOCK', 'TYPE_DECL', 'METHOD',
#     'METHOD_PARAMETER_IN', 'METHOD_PARAMETER_OUT', 'BLOCK', 'LOCAL',
#     'CALL', 'IDENTIFIER', 'LITERAL', 'CONTROL_STRUCTURE', 'RETURN',
#     'MODIFIER', 'TYPE_REF', 'BINDING', 'UNKNOWN_NODE'  # 最后的 'UNKNOWN_NODE' 用于安全处理未见类型
# ]
NODE_TYPE_MAP = {label: i for i, label in enumerate(NODE_LABELS)}

# 1.2 --- 新增 ---
# 为 CALL 节点的 DISPATCH_TYPE 创建映射
DISPATCH_TYPES = ['STATIC_DISPATCH', 'DYNAMIC_DISPATCH', 'UNKNOWN_DISPATCH']
DISPATCH_MAP = {label: i for i, label in enumerate(DISPATCH_TYPES)}

# 1.3 --- 新增 ---
# 为 CONTROL_STRUCTURE 节点的 CONTROL_STRUCTURE_TYPE 创建映射
CONTROL_TYPES = [
    'TRY', 'IF', 'ELSE', 'FOR', 'WHILE', 'DO', 'SWITCH', 'GOTO',
    'BREAK', 'CONTINUE', 'UNKNOWN_CONTROL'
]
CONTROL_MAP = {label: i for i, label in enumerate(CONTROL_TYPES)}


# 1.4 --- 修改 ---
# 辅助函数：编码单个节点的特征向量 (重写以包含更多结构特征)
def encode_node_features(node_data):
    """
    根据 GraphML 节点的属性生成固定维度的特征向量。

    Args:
        node_data (dict): NetworkX 节点属性字典。

    Returns:
        torch.Tensor: 节点的数值特征向量。
    """
    # 1. 节点类型 One-Hot 编码 (来自 cpg_dataset.py)
    node_label = node_data.get('labelV', 'UNKNOWN_NODE')
    type_id = NODE_TYPE_MAP.get(node_label, NODE_TYPE_MAP['UNKNOWN_NODE'])
    type_one_hot = torch.zeros(len(NODE_LABELS), dtype=torch.float)
    type_one_hot[type_id] = 1.0

    # 2. 数值特征 1: ORDER 属性 (来自 cpg_dataset.py)
    order = node_data.get('node__METHOD__ORDER',
                          node_data.get('node__LITERAL__ORDER',
                                        node_data.get('node__BLOCK__ORDER', 1)))
    order_feature = torch.tensor([float(order) / 100.0], dtype=torch.float)  # 归一化

    # 3. 数值特征 2: 代码文本长度 (来自 cpg_dataset.py)
    code_text = node_data.get('node__METHOD__CODE',
                              node_data.get('node__CALL__CODE',
                                            node_data.get('node__LOCAL__CODE',
                                                          node_data.get('node__BLOCK__CODE', ''))))
    code_length_feature = torch.tensor([len(str(code_text)) / 100.0], dtype=torch.float)  # 归一化

    # 4. --- 新增特征: 'IS_EXTERNAL' (布尔值) ---
    # 适用于 METHOD, TYPE_DECL 等节点
    is_external_str = node_data.get('node__METHOD__IS_EXTERNAL',
                                    node_data.get('node__TYPE_DECL__IS_EXTERNAL', 'false'))
    is_external_val = 1.0 if str(is_external_str).lower() == 'true' else 0.0
    is_external_feature = torch.tensor([is_external_val], dtype=torch.float)

    # 5. --- 新增特征: 'Line Span' (数值) ---
    # 计算代码行跨度
    start_line = node_data.get('node__METHOD__LINE_NUMBER',
                               node_data.get('node__TYPE_DECL__LINE_NUMBER', -1))
    end_line = node_data.get('node__METHOD__LINE_NUMBER_END',
                             node_data.get('node__TYPE_DECL__LINE_NUMBER_END', -1))
    span = 0.0
    try:
        if int(end_line) > 0 and int(start_line) > 0:
            span = float(int(end_line) - int(start_line) + 1)
    except ValueError:
        span = 0.0
    line_span_feature = torch.tensor([span / 100.0], dtype=torch.float)  # 归一化

    # 6. --- 新增特征: 'DISPATCH_TYPE' (One-Hot) ---
    # 仅适用于 CALL 节点
    dispatch_str = node_data.get('node__CALL__DISPATCH_TYPE', 'UNKNOWN_DISPATCH')
    dispatch_id = DISPATCH_MAP.get(dispatch_str, DISPATCH_MAP['UNKNOWN_DISPATCH'])
    dispatch_one_hot = torch.zeros(len(DISPATCH_TYPES), dtype=torch.float)
    dispatch_one_hot[dispatch_id] = 1.0

    # 7. --- 新增特征: 'CONTROL_STRUCTURE_TYPE' (One-Hot) ---
    # 仅适用于 CONTROL_STRUCTURE 节点
    control_str = node_data.get('node__CONTROL_STRUCTURE__CONTROL_STRUCTURE_TYPE', 'UNKNOWN_CONTROL')
    control_id = CONTROL_MAP.get(control_str, CONTROL_MAP['UNKNOWN_CONTROL'])
    control_one_hot = torch.zeros(len(CONTROL_TYPES), dtype=torch.float)
    control_one_hot[control_id] = 1.0

    # 8. 文本嵌入特征 (占位符) (来自 cpg_dataset.py)
    placeholder_feature = torch.zeros(16, dtype=torch.float)

    # 拼接最终特征向量
    return torch.cat((
        type_one_hot,  # 维度: len(NODE_LABELS)
        order_feature,  # 维度: 1
        code_length_feature,  # 维度: 1
        is_external_feature,  # 维度: 1
        line_span_feature,  # 维度: 1
        dispatch_one_hot,  # 维度: len(DISPATCH_TYPES)
        control_one_hot,  # 维度: len(CONTROL_TYPES)
        placeholder_feature  # 维度: 16
    ))


# ----------------------------------------------------
# 辅助函数：计算类别权重 (解决类别不平衡) (来自 cpg_dataset.py - 保持不变)
# ----------------------------------------------------
def get_class_weights(label_map, num_classes, effective_beta=0.9999):
    """
    计算类别权重，使用反向频率或 Eeffective Number of Samples 方法。
    """
    labels = list(label_map.values())
    if not labels:
        return torch.ones(num_classes)

    class_counts = np.bincount(labels, minlength=num_classes)

    # 避免除以零
    class_counts[class_counts == 0] = 1

    # 使用反向频率计算权重 (Simple Inverse Frequency)
    weights = 1.0 / class_counts

    # 归一化，使其和为类别数
    weights = weights / np.sum(weights) * num_classes

    # (可选) 使用 Class-Balanced Loss: effective_num = 1 - beta^n / (1 - beta)
    # effective_weights = (1 - effective_beta) / (1 - np.power(effective_beta, class_counts))
    # normalized_weights = effective_weights / np.sum(effective_weights) * num_classes

    return torch.tensor(weights, dtype=torch.float)


# =================================================================
# 2. PyTorch Geometric 自定义数据集类 (来自 cpg_dataset.py - 保持不变)
# =================================================================

class CPGGraphMLDataset(InMemoryDataset):
    """
    用于加载和转换 Joern GraphML 文件的自定义数据集。
    每个 GraphML 文件被视为图分类任务中的一个独立样本。
    """

    def __init__(self, root, graphml_dir, label_map, transform=None, pre_transform=None):
        """
        Args:
            root (str): 存储已处理数据的根目录。
            graphml_dir (str): 包含所有 GraphML 文件的目录路径。
            label_map (dict): 文件名到分类标签的映射字典 (例如: {'FileA.xml': 0, 'FileB.xml': 1})。
        """
        self.graphml_dir = graphml_dir
        self.label_map = label_map
        super().__init__(root, transform, pre_transform)
        # 加载预处理好的数据
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # 返回所有 GraphML 文件名列表
        return [os.path.basename(f) for f in glob(os.path.join(self.graphml_dir, "*.xml"))]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []

        # 遍历所有 GraphML 文件
        for raw_filename in tqdm(self.raw_file_names, desc="Processing GraphML files"):
            file_path = os.path.join(self.graphml_dir, raw_filename)
            file_base_name = raw_filename.rsplit('.', 1)[0]  # 移除 .xml 后缀

            # 1. 获取图的分类标签 (Y)
            class_label = self.label_map.get(file_base_name)
            if class_label is None:
                # 跳过没有标签的文件
                print(f"Skipping {raw_filename}: Label not found in map.")
                continue

            try:
                # 2. 使用 NetworkX 加载 GraphML 文件
                # NetworkX 可以直接读取 GraphML 文件
                nx_graph = nx.read_graphml(file_path)
            except Exception as e:
                print(f"Failed to load or parse GraphML {raw_filename}: {e}")
                continue

            # 3. 节点特征 (X) 和索引映射
            # 将 NetworkX 的节点 ID (Joern ID) 映射到 PyG 的 0-based 索引
            nx_nodes = list(nx_graph.nodes)
            if not nx_nodes:
                print(f"Warning: Graph for {raw_filename} has no nodes. Skipping.")
                continue

            node_map = {node_id: i for i, node_id in enumerate(nx_nodes)}

            # 生成节点特征矩阵 X
            features_list = [encode_node_features(nx_graph.nodes[node_id])
                             for node_id in nx_nodes]
            x = torch.stack(features_list)

            # 4. 边索引 (edge_index)
            edge_indices = []
            for u, v, _ in nx_graph.edges(data=True):
                # 确保边连接的节点都在当前子图内
                if u in node_map and v in node_map:
                    edge_indices.append([node_map[u], node_map[v]])

            if not edge_indices:
                # 如果图是空的或没有边，跳过此样本
                print(f"Warning: Graph for {raw_filename} has no edges. Skipping.")
                continue

            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

            # 5. 创建 PyTorch Geometric Data 对象
            data = Data(
                x=x,
                edge_index=edge_index,
                # 对于图分类任务，y 是图级别的标签
                y=torch.tensor([class_label], dtype=torch.long)
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # 6. 保存最终处理结果
        print(f"Successfully processed {len(data_list)} graphs.")
        torch.save(self.collate(data_list), self.processed_paths[0])


# --- (simple_mapping 和 main 函数保持不变) ---

def simple_mapping(folder_path, txt_file_path):
    """简化版本的映射函数（去掉.xml后缀）(来自 cpg_dataset.py - 保持不变)"""
    # 获取并排序xml文件
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.xml')],
        key=lambda x: int(re.search(r'(\d+)', x).group(1))
    )

    # 读取标签并转换为int
    with open(txt_file_path, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    # 统一为1
    trans_label = []
    for label in labels:
        if label != 0:
            label = 1
        trans_label.append(label)

    # 创建字典（去掉.xml后缀，标签为int）
    return {os.path.splitext(filename)[0]: label for filename, label in zip(files, trans_label)}


# =================================================================
# 3. 示例运行和数据使用 (来自 cpg_dataset.py - 保持不变)
# =================================================================

if __name__ == '__main__':
    # *** 1. 配置路径 ***
    ROOT_DIR = os.path.expanduser("~/project/postgraduate/model/GNN/")  # 根目录
    GRAPHML_INPUT_DIR = os.path.join(ROOT_DIR, "graphML/eval")  # 包含 GraphML 文件的目录
    LABEL_DIR = os.path.join(ROOT_DIR, "labels/eval_labels.txt")
    PROCESSED_DATA_ROOT = os.path.join(ROOT_DIR, "processed_dataset")  # 存储最终 .pt 文件的目录

    # *** 2. 准备标签映射 ***
    # 假设您的 GraphML 文件名为 MethodWrapper0178.xml，标签是 1
    # 您需要根据您的实际分类任务构建这个字典。
    # 这里我们模拟一个标签映射：
    LABEL_MAPPING = simple_mapping(GRAPHML_INPUT_DIR, LABEL_DIR)

    # 确保您的文件名在 LABEL_MAPPING 中是 *没有扩展名* 的基础文件名

    print("Starting data processing...")

    # 实例化自定义数据集，这将触发 process() 方法进行转换
    dataset = CPGGraphMLDataset(
        root=PROCESSED_DATA_ROOT,
        graphml_dir=GRAPHML_INPUT_DIR,
        label_map=LABEL_MAPPING
    )

    print(f"\n--- Dataset Summary ---")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of node features: {dataset.num_node_features}")  # 这将自动反映新的维度
    print(f"Number of classes (if any): {dataset.num_classes}")

    # 示例: 访问第一个图数据点
    first_graph = dataset[0]
    print(f"\nFirst Data object: {first_graph}")
    print(f"Node feature shape (x): {first_graph.x.shape}")
    print(f"Edge index shape (edge_index): {first_graph.edge_index.shape}")

    # --- 然后您可以像标准的 PyG 数据集一样使用它进行训练 ---
    # from torch_geometric.loader import DataLoader
    # loader = DataLoader(dataset, batch_size=32, shuffle=True)