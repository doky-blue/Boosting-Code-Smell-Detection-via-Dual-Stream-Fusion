import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from datetime import datetime
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, global_max_pool, GATConv
from torch_geometric.loader import DataLoader
from cpg_dataset import CPGGraphMLDataset, simple_mapping, get_class_weights
# --- 修改 1: 导入 roc_auc_score ---
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score  # <--- 新增
)

# 设置随机种子以保证结果可复现
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


# ----------------------------------------------------
# 0. 日志记录类
# ----------------------------------------------------
class DualLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')
        self.encoding = 'utf-8'

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ----------------------------------------------------
# 0.5 二分类 Focal Loss
# ----------------------------------------------------
class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt).pow(self.gamma)
        loss = focal_term * bce_loss

        if self.alpha is not None:
            weights = self.alpha * targets + (1.0 - targets)
            loss = loss * weights

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# ----------------------------------------------------
# 1. 模型定义
# ----------------------------------------------------

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x


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
# 2. 训练和测试函数
# ----------------------------------------------------

def train(model, device, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze(1)
        target = data.y.float()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        predicted_probs = torch.sigmoid(out)
        pred = (predicted_probs > 0.5).long()
        correct += (pred == data.y).sum().item()
        total_samples += data.num_graphs

    return total_loss / total_samples, correct / total_samples


@torch.no_grad()
def find_best_threshold(model, device, loader):
    model.eval()
    all_probs = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        out = model(data).squeeze(1)
        predicted_probs = torch.sigmoid(out)
        all_probs.extend(predicted_probs.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)

    try:
        best_f1_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
    except ValueError:
        best_threshold = 0.5
        best_f1 = 0.0

    return best_threshold, best_f1


@torch.no_grad()
def evaluate_metrics(model, device, loader, criterion, num_classes, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # 用于 AUC
    total_loss = 0
    total_samples = 0

    for data in loader:
        data = data.to(device)
        out = model(data).squeeze(1)
        target = data.y.float()

        total_loss += criterion(out, target).item() * data.num_graphs
        total_samples += data.num_graphs

        predicted_probs = torch.sigmoid(out)
        preds = (predicted_probs > threshold).long()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())
        all_probs.extend(predicted_probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)

    # --- 计算指标 ---

    # 1. 基础指标
    overall_accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)

    # 2. MCC
    mcc = matthews_corrcoef(y_true, y_pred)

    # 3. AUC (Area Under Curve) - 需要概率值
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0  # 处理只有一个类别的情况

    # 4. G-mean (Geometric Mean) = sqrt(Sensitivity * Specificity)
    # 强制 labels=[0, 1] 确保返回 2x2 矩阵，即使预测结果全为某一类
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    g_mean = np.sqrt(sensitivity * specificity)

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=[f'Class {i}' for i in range(num_classes)],
        output_dict=True,
        zero_division=0
    )

    results = {
        'loss': total_loss / total_samples,
        'overall_accuracy': overall_accuracy,
        'weighted_f1': f1,
        'weighted_recall': recall,
        'precision': precision,
        'conf_matrix': conf_matrix,
        'mcc': mcc,
        'auc': auc,  # <--- 新增
        'g_mean': g_mean,  # <--- 新增
        'per_class_report': report_dict
    }

    return results


# ----------------------------------------------------
# 3. 数据加载和主程序
# ----------------------------------------------------

def main(smell):
    original_stdout = sys.stdout
    ROOT_DIR = os.path.expanduser("~/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn")
    NUM_CLASSES = 2
    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE_PATH = os.path.join(LOG_DIR, f"training_log_{smell}.txt")

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

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

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

        if len(train_dataset) == 0:
            print(f"Error: No data found for smell '{smell}' in training set. Skipping.")
            return

        CURRENT_NODE_FEATURE_DIM = train_dataset.num_node_features
        print(f"Detected Node Feature Dimension: {CURRENT_NODE_FEATURE_DIM}")

        BATCH_SIZE = 32
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        class_weights = get_class_weights(TRAIN_LABEL_MAPPING, NUM_CLASSES)
        print(f"Calculated Class Weights: {class_weights}")

        num_pos_samples = np.sum(list(TRAIN_LABEL_MAPPING.values()))
        total_samples = len(TRAIN_LABEL_MAPPING)
        pos_weight = (total_samples - num_pos_samples) / num_pos_samples if num_pos_samples > 0 else 1.0

        model = GAT_JK_Pool(in_channels=CURRENT_NODE_FEATURE_DIM,
                            hidden_channels=128,
                            out_channels=1).to(device)

        alpha_tensor = torch.tensor(pos_weight).to(device)
        criterion = BinaryFocalLoss(alpha=alpha_tensor, gamma=1.0).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=20, verbose=True
        )

        print(f"\nModel initialized:\n{model}")
        print(
            f"Training on {len(train_dataset)} graphs, validation on {len(eval_dataset)} graphs, testing on {len(test_dataset)} graphs.")

        EPOCHS = 300
        best_eval_f1 = 0.0
        best_threshold_for_model = 0.5

        date = datetime.now().strftime("%Y-%m-%d")
        MODEL_SAVE_DIR = f"/home/doky/project/postgraduate/model/GNN/saved_models/{smell}/{date}/focal_loss"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "gcn_cpg_model_best.pt")

        print(f"Best model will be saved to: {MODEL_SAVE_PATH}\n")

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
            current_best_threshold, current_eval_f1 = find_best_threshold(model, device, eval_loader)
            eval_results = evaluate_metrics(model, device, eval_loader, criterion, NUM_CLASSES,
                                            threshold=current_best_threshold)

            eval_loss = eval_results['loss']
            eval_acc = eval_results['overall_accuracy']

            if current_eval_f1 > best_eval_f1:
                best_eval_f1 = current_eval_f1
                best_threshold_for_model = current_best_threshold
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(
                    f"Epoch {epoch:03d}: New best eval F1: {best_eval_f1:.4f} at threshold {best_threshold_for_model:.4f}. Model saved.")

            scheduler.step(current_eval_f1)

            if epoch % 10 == 0 or epoch == EPOCHS:
                # 日志中也可以展示更多信息
                print(f'Epoch: {epoch:03d}, '
                      f'Train Loss: {train_loss:.4f}, '
                      f'Eval Loss: {eval_loss:.4f}, Eval F1: {current_eval_f1:.4f}, '
                      f'G-mean: {eval_results["g_mean"]:.4f}')

        print("\n--- Final Evaluation on Test Set (Loading Best Model) ---")
        print(f"Using optimal threshold found on validation set: {best_threshold_for_model:.4f}")

        best_model = GAT_JK_Pool(in_channels=CURRENT_NODE_FEATURE_DIM, hidden_channels=128, out_channels=1).to(device)
        best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        final_test_results = evaluate_metrics(best_model, device, test_loader, criterion, NUM_CLASSES,
                                              threshold=best_threshold_for_model)

        print(f'Final Test Loss: {final_test_results["loss"]:.4f}')
        print(f'Final Test Overall Accuracy: {final_test_results["overall_accuracy"]:.4f}')

        # --- 修改: 打印核心指标 ---
        print("-" * 20 + " Core Metrics " + "-" * 20)
        print(f'F1-Score (Pos) : {final_test_results["weighted_f1"]:.4f}')
        print(f'MCC            : {final_test_results["mcc"]:.4f}')
        print(f'AUC            : {final_test_results["auc"]:.4f}')
        print(f'G-mean         : {final_test_results["g_mean"]:.4f}')
        print("-" * 54)

        print("\n--- Final Test Per-Class Metrics ---")
        report_df = pd.DataFrame(final_test_results['per_class_report']).transpose()
        print(report_df.to_markdown(floatfmt=".4f"))

        print("\nFinal Test Confusion Matrix:")
        print(final_test_results['conf_matrix'])
        print(f"--- Finished training for smell: {smell} ---")

    finally:
        if 'logger' in locals():
            sys.stdout = original_stdout
            logger.close()
            print(f"\nLog for '{smell}' saved to {LOG_FILE_PATH}")


if __name__ == '__main__':
    smells = ["blob", "data_class", "feature_envy", "long_method"]
    original_stdout_main = sys.stdout

    for smell in smells:
        main(smell)
        sys.stdout = original_stdout_main