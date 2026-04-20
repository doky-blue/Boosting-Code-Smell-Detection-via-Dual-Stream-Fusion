import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm, global_max_pool

# --- 导入你的数据集处理类 ---
# 确保 cpg_dataset.py 在同一目录下，或者在 PYTHONPATH 中
from cpg_dataset import CPGGraphMLDataset, simple_mapping
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_score,
    matthews_corrcoef
)


# ----------------------------------------------------
# 1. 模型定义 (必须与训练代码完全一致)
# ----------------------------------------------------
class GAT_JK_Pool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.dropout_p = 0.4
        heads = 4

        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=self.dropout_p)
        self.bn1 = BatchNorm(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout_p)
        self.bn2 = BatchNorm(hidden_channels * heads)

        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=self.dropout_p)
        self.bn3 = BatchNorm(hidden_channels)

        lin_in_features = (hidden_channels * heads * 2) + \
                          (hidden_channels * heads * 2) + \
                          (hidden_channels * 2)

        self.lin1 = torch.nn.Linear(lin_in_features, hidden_channels)
        self.lin_out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.1)
        x1_pool = torch.cat([global_mean_pool(x1, batch), global_max_pool(x1, batch)], dim=1)
        x1 = F.dropout(x1, p=self.dropout_p, training=self.training)

        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.1)
        x2_pool = torch.cat([global_mean_pool(x2, batch), global_max_pool(x2, batch)], dim=1)
        x2 = F.dropout(x2, p=self.dropout_p, training=self.training)

        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.1)
        x3_pool = torch.cat([global_mean_pool(x3, batch), global_max_pool(x3, batch)], dim=1)

        x = torch.cat([x1_pool, x2_pool, x3_pool], dim=1)

        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.lin_out(x)
        return x


# ----------------------------------------------------
# 2. 评估函数 (简化版，无需 Loss)
# ----------------------------------------------------
@torch.no_grad()
def evaluate_model(model, device, loader, num_classes, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    print(f"Starting evaluation with threshold: {threshold}...")

    for data in loader:
        data = data.to(device)
        out = model(data).squeeze(1)

        # 计算概率
        predicted_probs = torch.sigmoid(out)

        # 应用阈值
        preds = (predicted_probs > threshold).long()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_probs.extend(predicted_probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # --- 计算指标 ---
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=[f'Class {i}' for i in range(num_classes)],
        output_dict=True,
        zero_division=0
    )

    return {
        'mcc': mcc,
        'f1': f1,
        'conf_matrix': conf_matrix,
        'report': report_dict
    }


# ----------------------------------------------------
# 3. 主程序
# ----------------------------------------------------
def main(smell_type, threshold=0.5):
    # --- 配置路径 ---
    # 请根据实际情况修改 ROOT_DIR
    ROOT_DIR = os.path.expanduser("~/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn")
    MODEL_DIR = f"/home/doky/project/postgraduate/model/GNN/saved_models/{smell_type}"
    MODEL_PATH = os.path.join(MODEL_DIR, "gcn_cpg_model_best.pt")

    # 测试集路径
    TEST_INPUT_DIR = os.path.join(ROOT_DIR, "graphML", smell_type, "test")
    TEST_LABEL = os.path.join(ROOT_DIR, "labels", smell_type, "test", "labels.txt")
    TEST_PROCESSED_DATA_ROOT = os.path.join(ROOT_DIR, "processed_dataset", smell_type, "test")

    print(f"--- Evaluating Smell: {smell_type} ---")
    print(f"Model Path: {MODEL_PATH}")

    # 检查模型是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 加载数据 ---
    # 获取标签映射
    test_label_mapping = simple_mapping(TEST_INPUT_DIR, TEST_LABEL)

    test_dataset = CPGGraphMLDataset(
        root=TEST_PROCESSED_DATA_ROOT,
        graphml_dir=TEST_INPUT_DIR,
        label_map=test_label_mapping
    )

    if len(test_dataset) == 0:
        print("Error: Test dataset is empty.")
        return

    # 自动获取特征维度 (必须与训练时一致，通常test和train特征维度相同)
    num_node_features = test_dataset.num_node_features
    print(f"Detected Node Feature Dimension: {num_node_features}")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # --- 加载模型 ---
    model = GAT_JK_Pool(in_channels=num_node_features, hidden_channels=128, out_channels=1)

    # 加载权重
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    model.to(device)

    # --- 运行评估 ---
    results = evaluate_model(model, device, test_loader, num_classes=2, threshold=threshold)

    # --- 打印结果 ---
    print("\n" + "=" * 40)
    print(f"Evaluation Results for [{smell_type}]")
    print("=" * 40)
    print(f"Threshold Used: {threshold}")
    print(f"MCC: {results['mcc']:.4f}")
    print(f"F1-Score:       {results['f1']:.4f}")

    print("\nConfusion Matrix:")
    print(results['conf_matrix'])

    print("\nPer-Class Report:")
    report_df = pd.DataFrame(results['report']).transpose()
    print(report_df.to_markdown(floatfmt=".4f"))
    print("\n")


if __name__ == "__main__":
    smells = ['blob', 'data_class', 'feature_envy', 'long_method']
    thr_dict = {
        'blob': 0.4705,
        'data_class': 0.4838,
        'feature_envy': 0.4966,
        'long_method': 0.4948,
    }

    for smell in smells:
        main(smell, thr_dict[smell])