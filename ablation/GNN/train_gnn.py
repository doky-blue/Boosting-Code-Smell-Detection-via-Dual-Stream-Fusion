import os
import sys  # --- 新增 ---
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, global_max_pool, GATConv
from torch_geometric.loader import DataLoader
# --- 修改：移除了 NODE_FEATURE_DIM ---
from cpg_dataset import CPGGraphMLDataset, simple_mapping, get_class_weights
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_score,
    matthews_corrcoef,
    precision_recall_curve  # <--- 新增导入
)

# 设置随机种子以保证结果可复现
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


# ----------------------------------------------------
# 0. --- 新增：双重日志记录类 ---
# ----------------------------------------------------
class DualLogger(object):
    """
    一个辅助类，用于将所有 print 语句 (stdout) 同时写入
    控制台和指定的日志文件。
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')
        self.encoding = 'utf-8'  # 确保

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        # flush 方法对于确保实时写入文件很重要
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ----------------------------------------------------
# 0.5 --- 新增：二分类 Focal Loss ---
# ----------------------------------------------------
class BinaryFocalLoss(torch.nn.Module):
    """
    适配二分类任务 (out_channels=1) 的 Focal Loss。
    替换原本的 BCEWithLogitsLoss。
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor or float, optional): 正样本的权重 (即 pos_weight)。
            gamma (float): 聚焦参数，默认 2.0。
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (Batch_Size,)  <-- 你的模型输出
        # targets: (Batch_Size,) <-- 你的标签

        # 1. 计算标准的 BCE Loss (不进行归约，保留每个样本的 Loss)
        # logit 还没有经过 sigmoid，所以用 binary_cross_entropy_with_logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # 2. 计算 pt (预测概率与真实标签的一致性)
        # 对于二分类，pt = exp(-bce_loss) 在数学上是成立的
        pt = torch.exp(-bce_loss)

        # 3. 计算 Focal Term: (1 - pt)^gamma
        focal_term = (1 - pt).pow(self.gamma)

        # 4. 基础 Loss
        loss = focal_term * bce_loss

        # 5. 应用 Alpha (作为 pos_weight 处理)
        if self.alpha is not None:
            # 如果 alpha 是 pos_weight，它只应该增强正样本 (target==1) 的权重
            # 这种实现方式模仿 BCEWithLogitsLoss 的 pos_weight 行为
            # 构造权重向量：正样本乘 alpha，负样本乘 1.0
            weights = self.alpha * targets + (1.0 - targets)
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ----------------------------------------------------
# 1. 模型定义：GCN (保持不变)
# ----------------------------------------------------

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)  # 添加 BatchNorm

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)  # 添加 BatchNorm

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)  # 添加 BatchNorm

        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. GCN 层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  # 应用 BatchNorm
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=0.4, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)  # 应用 BatchNorm
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=0.4, training=self.training)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)  # 应用 BatchNorm
        x = F.leaky_relu(x, negative_slope=0.1)

        # 2. 全局平均池化
        x = global_mean_pool(x, batch)  # 确保使用 torch_geometric.nn.global_mean_pool

        # 3. 线性分类层
        x = self.lin(x)
        return x


# --- GAT_JK_Pool (保持不变) ---
class GAT_JK_Pool(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.dropout_p = 0.4  # 保持与 GCN 一致的 dropout
        heads = 4  # GAT 注意力头的数量

        # --- GAT 层 ---
        # GATConv 的输出维度是 (hidden_channels * heads)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=self.dropout_p)
        self.bn1 = BatchNorm(hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=self.dropout_p)
        self.bn2 = BatchNorm(hidden_channels * heads)

        # 最后一层 GAT，使用 1 个头并禁用拼接 (concat=False)
        self.conv3 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=self.dropout_p)
        self.bn3 = BatchNorm(hidden_channels)

        # --- "Jumping Knowledge" + "Dual Pooling" 线性层 ---
        # 我们将从3个层、每层2种池化（mean + max）中收集特征
        # L1: (hidden * heads) * 2
        # L2: (hidden * heads) * 2
        # L3: (hidden) * 2

        # 你的 GCN 中 hidden_channels = 128
        # L1_dim = (128 * 4) * 2 = 1024
        # L2_dim = (128 * 4) * 2 = 1024
        # L3_dim = (128 * 1) * 2 = 256
        # total_dim = 1024 + 1024 + 256 = 2304

        lin_in_features = (hidden_channels * heads * 2) + \
                          (hidden_channels * heads * 2) + \
                          (hidden_channels * 2)

        # 定义一个小型 MLP 作为分类器，以处理拼接后的高维特征
        self.lin1 = torch.nn.Linear(lin_in_features, hidden_channels)
        self.lin_out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # --- Layer 1 ---
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.1)
        x1_pool = torch.cat([global_mean_pool(x1, batch), global_max_pool(x1, batch)], dim=1)
        x1 = F.dropout(x1, p=self.dropout_p, training=self.training)

        # --- Layer 2 ---
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.1)
        x2_pool = torch.cat([global_mean_pool(x2, batch), global_max_pool(x2, batch)], dim=1)
        x2 = F.dropout(x2, p=self.dropout_p, training=self.training)

        # --- Layer 3 ---
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.1)
        x3_pool = torch.cat([global_mean_pool(x3, batch), global_max_pool(x3, batch)], dim=1)

        # --- 全局读出 (Readout) ---
        # 拼接所有层的池化结果 (Jumping Knowledge)
        x = torch.cat([x1_pool, x2_pool, x3_pool], dim=1)

        # --- 线性分类层 ---
        x = self.lin1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.lin_out(x)
        return x


# ----------------------------------------------------
# 2. 训练和测试函数 (保持不变)
# ----------------------------------------------------

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data).squeeze(1)  # 模型输出是 (Batch_Size, 1)，压缩成 (Batch_Size,)

        # 标签转换：BCEWithLogitsLoss 要求标签为浮点数
        # data.y 必须为 float 且形状为 (Batch_Size,)
        target = data.y.float()

        loss = criterion(out, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        # 计算准确率
        # 计算准确率：使用 Sigmoid 激活，阈值 0.5
        predicted_probs = torch.sigmoid(out)
        pred = (predicted_probs > 0.5).long()  # 得到 0 或 1 的预测值

        correct += (pred == data.y).sum().item()
        total_samples += data.num_graphs

    return total_loss / total_samples, correct / total_samples

@torch.no_grad()
def find_best_threshold(model, device, loader):
    """
    在给定的 loader (通常是验证集) 上寻找最大化 F1-score 的最佳阈值。
    """
    model.eval()
    all_probs = []
    all_labels = []

    # 1. 收集所有真实标签和预测概率
    for data in loader:
        data = data.to(device)
        out = model(data).squeeze(1)  # (Batch_Size,)
        predicted_probs = torch.sigmoid(out)

        all_probs.extend(predicted_probs.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)

    # 2. 计算精确率-召回率曲线
    # precision[i], recall[i] 对应于 thresholds[i]
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    # 3. 计算所有阈值的 F1-score
    # 我们使用 f1 = 2 * (P * R) / (P + R)，并避免除以零
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)

    # 4. 找到最佳F1对应的阈值
    # precision_recall_curve 返回的 P/R 数组比 thresholds 数组多一个元素
    # 我们只关心 f1_scores[:-1] 对应的 thresholds
    try:
        best_f1_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
    except ValueError:
        # 如果验证集为空或只有一个类别，则回退
        best_threshold = 0.5
        best_f1 = 0.0

    return best_threshold, best_f1


@torch.no_grad()
def evaluate_metrics(model, device, loader, criterion, num_classes, threshold=0.5):  # <--- 1. 添加 threshold 参数, 默认 0.5
    # 保持模型评估模式和数据收集部分不变
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        out = model(data).squeeze(1)  # 压缩成 (Batch_Size,)

        target = data.y.float()

        total_loss += criterion(out, target).item() * data.num_graphs
        total_samples += data.num_graphs

        predicted_probs = torch.sigmoid(out)

        # --- 2. 使用传入的 threshold ---
        preds = (predicted_probs > threshold).long()  # <--- 修改此行

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    # 转换为 NumPy 数组
    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    # --- 计算全局指标 ---

    # 整体准确率
    overall_accuracy = accuracy_score(y_true, y_pred)

    # 加权平均指标 (用于整体评估)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # 混淆矩阵
    conf_matrix = confusion_matrix(y_true, y_pred)

    # --- 计算按类别划分的指标 (核心需求) ---

    # 使用 classification_report 获取按类别划分的精确率、召回率和 F1-score
    # output_dict=True 将结果以字典形式返回，方便后续处理
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),  # 确保包含所有可能的类别
        target_names=[f'Class {i}' for i in range(num_classes)],
        output_dict=True,
        zero_division=0
    )

    results = {
        'loss': total_loss / total_samples,
        'overall_accuracy': overall_accuracy,
        'weighted_f1': f1,
        'mcc': mcc,
        'weighted_recall': recall,
        'precision': precision,
        'conf_matrix': conf_matrix,
        'per_class_report': report_dict
    }

    return results


# ----------------------------------------------------
# 3. 数据加载和主程序
# ----------------------------------------------------

def main(smell):
    # --- 新增：保存原始的 stdout ---
    original_stdout = sys.stdout

    # --- 配置参数 ---
    # 使用您的实际路径
    ROOT_DIR = os.path.expanduser("~/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn")
    NUM_CLASSES = 2

    # --- 新增：定义日志文件路径 ---
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE_PATH = os.path.join(LOG_DIR, f"training_log_{smell}.txt")

    # --- 新增：设置 stdout 重定向 ---
    # 使用 try...finally 确保无论是否出错，stdout 都会被恢复
    logger = DualLogger(LOG_FILE_PATH)
    sys.stdout = logger

    try:
        print(f"--- Starting training for smell: {smell} ---")
        print(f"Log file will be saved to: {LOG_FILE_PATH}\n")

        TRAIN_INPUT_DIR = os.path.join(ROOT_DIR, "graphML", smell, "train")
        TRAIN_LABEL = os.path.join(ROOT_DIR, "labels", smell, "train", "labels.txt")
        TRAIN_PROCESSED_DATA_ROOT = os.path.join(ROOT_DIR, "processed_dataset", smell, "train")

        EVAL_INPUT_DIR = os.path.join(ROOT_DIR, "graphML", smell, "eval")
        EVAL_LABEL = os.path.join(ROOT_DIR, "labels", smell, "eval", "labels.txt")
        EVAL_PROCESSED_DATA_ROOT = os.path.join(ROOT_DIR, "processed_dataset", smell, "eval")

        TEST_INPUT_DIR = os.path.join(ROOT_DIR, "graphML", smell, "test")
        TEST_LABEL = os.path.join(ROOT_DIR, "labels", smell, "test", "labels.txt")
        TEST_PROCESSED_DATA_ROOT = os.path.join(ROOT_DIR, "processed_dataset", smell, "test")

        EVAL_LABEL_MAPPING = simple_mapping(EVAL_INPUT_DIR, EVAL_LABEL)
        TRAIN_LABEL_MAPPING = simple_mapping(TRAIN_INPUT_DIR, TRAIN_LABEL)
        TEST_LABEL_MAPPING = simple_mapping(TEST_INPUT_DIR, TEST_LABEL)

        # --- 设备设置 ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # --- 1. 加载数据 ---
        train_dataset = CPGGraphMLDataset(
            root=TRAIN_PROCESSED_DATA_ROOT,
            graphml_dir=TRAIN_INPUT_DIR,
            label_map={k: v for k, v in TRAIN_LABEL_MAPPING.items()}
        )

        eval_dataset = CPGGraphMLDataset(
            root=EVAL_PROCESSED_DATA_ROOT,
            graphml_dir=EVAL_INPUT_DIR,
            label_map={k: v for k, v in EVAL_LABEL_MAPPING.items()}
        )

        test_dataset = CPGGraphMLDataset(
            root=TEST_PROCESSED_DATA_ROOT,
            graphml_dir=TEST_INPUT_DIR,
            label_map={k: v for k, v in TEST_LABEL_MAPPING.items()}
        )

        # --- 修复：在加载数据集后获取 num_node_features ---
        # 确保数据集至少有一个图
        if len(train_dataset) == 0:
            print(f"Error: No data found for smell '{smell}' in training set. Skipping.")
            return  # 退出此smell的main函数

        CURRENT_NODE_FEATURE_DIM = train_dataset.num_node_features
        print(f"Detected Node Feature Dimension: {CURRENT_NODE_FEATURE_DIM}")

        # 创建 DataLoader (对于 GNN 图分类任务，通常使用 DataLoader 进行批量处理)
        BATCH_SIZE = 32
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # --- 2. 计算和应用类别权重 ---
        class_weights = get_class_weights(TRAIN_LABEL_MAPPING, NUM_CLASSES)
        print(f"Calculated Class Weights: {class_weights}")  # 检查权重是否正确（少数类权重更高）

        # --- 3. 初始化模型、优化器和损失函数 ---

        # 应用类别权重到损失函数
        num_pos_samples = np.sum(list(TRAIN_LABEL_MAPPING.values()))
        total_samples = len(TRAIN_LABEL_MAPPING)
        pos_weight = (total_samples - num_pos_samples) / num_pos_samples if num_pos_samples > 0 else 1.0

        # **模型输出 (out_channels) 必须从 2 改回 1**
        # model = GCN(in_channels=CURRENT_NODE_FEATURE_DIM, # <-- 修复
        #             hidden_channels=128,
        #             out_channels=1).to(device)  # 输出通道改为 1

        # 使用新的 GAT_JK_Pool 模型
        # --- 修复：使用 CURRENT_NODE_FEATURE_DIM ---
        model = GAT_JK_Pool(in_channels=CURRENT_NODE_FEATURE_DIM,
                            hidden_channels=128,
                            out_channels=1).to(device)  # 同样是1个输出通道

        # 损失函数使用 pos_weight 处理类别不平衡
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))

        # --- 修改：使用 BinaryFocalLoss ---
        # 将计算出的 pos_weight 转换为 Tensor
        # alpha_tensor = torch.tensor(pos_weight).to(device)

        # 初始化 Focal Loss
        # alpha: 传入 pos_weight
        # gamma: 通常设为 2.0，你可以尝试 1.5 到 3.0 之间的值
        # criterion = BinaryFocalLoss(alpha=alpha_tensor, gamma=1.0).to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=5e-4,  # 降低学习率以稳定训练
                                     weight_decay=1e-3)  # 显著降低 L2 正则化

        # 新增：添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',  # 我们希望 F1-score 最大化
            factor=0.1,  # 当F1停滞时，学习率乘以 0.1
            patience=20,  # 20 个 epoch F1-score没有提升则触发
            verbose=True
        )

        print(f"\nModel initialized:\n{model}")
        print(
            f"Training on {len(train_dataset)} graphs, validation on {len(eval_dataset)} graphs, testing on {len(test_dataset)} graphs.")

        # --- 4. 训练循环 (修改调用) ---
        EPOCHS = 300
        best_eval_f1 = 0.0  # --- 修改：根据验证集F1保存模型 ---
        best_threshold_for_model = 0.5

        # --- 修改：定义模型保存路径 ---
        date = datetime.now().strftime("%Y-%m-%d")
        MODEL_SAVE_DIR = f"/home/doky/project/postgraduate/model/GNN/saved_models/{smell}/{date}/bce_loss"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "gcn_cpg_model_best.pt")

        print(f"Best model will be saved to: {MODEL_SAVE_PATH}\n")

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)

            # --- 步骤 1: 寻找当前模型在验证集上的最佳阈值和对应的F1 ---
            current_best_threshold, current_eval_f1 = find_best_threshold(model, device, eval_loader)

            # --- 步骤 2: 使用该阈值获取验证集上的所有其他指标 (如 loss, acc) ---
            eval_results = evaluate_metrics(model, device, eval_loader, criterion, NUM_CLASSES,
                                            threshold=current_best_threshold)

            eval_loss = eval_results['loss']
            eval_acc = eval_results['overall_accuracy']

            # --- 2. 早停逻辑 (基于验证集F1) ---
            if current_eval_f1 > best_eval_f1:
                best_eval_f1 = current_eval_f1
                best_threshold_for_model = current_best_threshold  # <--- 保存最佳阈值

                # 在最佳模型性能时保存模型
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(
                    f"Epoch {epoch:03d}: New best eval F1: {best_eval_f1:.4f} at threshold {best_threshold_for_model:.4f}. Model saved.")  # <--- 修改日志

            # 在评估后，使用验证集F1-score更新调度器
            scheduler.step(current_eval_f1)

            if epoch % 10 == 0 or epoch == EPOCHS:
                print(f'Epoch: {epoch:03d}, '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.4f}, Eval F1: {current_eval_f1:.4f}, Best Thr: {current_best_threshold:.4f}')  # <--- 修改日志

            # --- 5. 最终评估 (修改调用和输出) ---
        print("\n--- Final Evaluation on Test Set (Loading Best Model) ---")

        # <--- 新增: 打印用于测试的阈值 ---
        print(f"Using optimal threshold found on validation set: {best_threshold_for_model:.4f}")

        # --- 修复：确保这里也使用 CURRENT_NODE_FEATURE_DIM ---
        best_model = GAT_JK_Pool(in_channels=CURRENT_NODE_FEATURE_DIM, hidden_channels=128, out_channels=1).to(device)
        best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        # --- 3. 修改：在测试集 (test_loader) 上运行最终评估 ---
        # <--- 将保存的最佳阈值 (best_threshold_for_model) 传递给评估函数 ---
        final_test_results = evaluate_metrics(best_model, device, test_loader, criterion, NUM_CLASSES,
                                              threshold=best_threshold_for_model)

        print(f'Final Test Loss: {final_test_results["loss"]:.4f}')
        print(f'Final Test Overall Accuracy: {final_test_results["overall_accuracy"]:.4f}')
        print(f'Final Test Weighted F1-Score: {final_test_results["weighted_f1"]:.4f}')
        print(f'Final Test MCC: {final_test_results["mcc"]:.4f}')

        print("\n--- Final Test Per-Class Metrics ---")
        # 将报告字典转换为 Pandas DataFrame 以获得美观的表格输出
        report_df = pd.DataFrame(final_test_results['per_class_report']).transpose()
        print(report_df.to_markdown(floatfmt=".4f"))

        print("\nFinal Test Confusion Matrix:")
        print(final_test_results['conf_matrix'])
        print(f"--- Finished training for smell: {smell} ---")

    finally:
        # --- 新增：恢复 stdout 并关闭文件 ---
        if 'logger' in locals():
            sys.stdout = original_stdout
            logger.close()
            # 在控制台打印一条最终消息
            print(f"\nLog for '{smell}' saved to {LOG_FILE_PATH}")


if __name__ == '__main__':
    smells = ["blob", "data_class", "feature_envy", "long_method"]
    original_stdout_main = sys.stdout  # 保存主循环开始前的stdout

    for smell in smells:
        main(smell)
        sys.stdout = original_stdout_main  # 确保每次循环后都恢复