import json
import torch
import os
import numpy as np
from torch.functional import F
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from peft import PeftModel
from xgboost import XGBClassifier

# 修改 1: 导入 roc_auc_score 和 confusion_matrix
from sklearn.metrics import classification_report, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from torch_geometric.loader import DataLoader
from train_gnn import GAT_JK_Pool
from cpg_dataset import CPGGraphMLDataset, simple_mapping


@torch.no_grad()
def codebert_predict(texts, model, tokenizer, device, batch_size=16):
    model.eval()
    preds = []
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits  # [B, 2]
        logit_probs = F.softmax(logits, dim=1)  # 形状仍为 [B, 2]
        preds.append(logits.argmax(1).cpu())
        all_logits.append(logit_probs.cpu())
    return torch.cat(preds).numpy(), torch.cat(all_logits).numpy()


def gcn_predict(model, device, loader, threshold=0.70):
    model.eval()  # 1. 切换到评估模式 (重要)
    all_preds = []
    all_probs = []  # <--- 新增：用于存储概率的列表

    # 2. 评估时不需要计算梯度 (重要)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze(1)  # 压缩成 (Batch_Size,)

            # 3. 计算概率 (你原有的代码，是正确的)
            # sigmoid 适用于二分类的单个 logit 输出
            predicted_probs = torch.sigmoid(out)  # 形状 (B,)

            # 4. 根据阈值计算预测标签
            preds = (predicted_probs > threshold).long()

            # 5. 收集标签和概率
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(predicted_probs.cpu().numpy())  # <--- 关键改动

    # 将列表转换为 NumPy 数组
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    # 返回预测标签 (基于阈值) 和 概率
    return y_pred, y_probs


def merge_res(codebert_preds, gcn_preds):
    res_preds = []
    for i, j in zip(codebert_preds, gcn_preds):
        if i == 0 and j == 0:
            res_preds.append(0)
        else:
            res_preds.append(1)
    return res_preds


def weight_merge(codebert_preds, gcn_preds, weight=0.6, threshold=0.40):
    res_preds = []

    for i, j in zip(codebert_preds, gcn_preds):
        if (i * weight + j * (1 - weight)) > threshold:
            res_preds.append(1)
        else:
            res_preds.append(0)

        # res_preds.append((pred > threshold).int())

    return res_preds


PATHS = {
    "base_model": "/home/doky/Llama/codebert-base/",
    "peft_root": "/home/doky/project/postgraduate/model/final_model/codebert/{}/",
    "test_file": "/home/doky/project/postgraduate/dataset/MLCQ/data/{}/{}_test_four_model.jsonl",
    "eval_file": "/home/doky/project/postgraduate/dataset/MLCQ/data/{}/{}_eval_four_model.jsonl",
    "save_dir": "/home/doky/project/postgraduate/eval/four_model_fix/",
}


def eval():
    smells = ["blob", "data_class", "feature_envy", "long_method"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(PATHS["base_model"])

    # codebert预测
    for smell in smells:
        print("smell:" + smell)
        # 5.1 加载模型
        base = RobertaForSequenceClassification.from_pretrained(
            PATHS["base_model"], num_labels=2)
        peft_path = PATHS["peft_root"].format(smell)
        model = PeftModel.from_pretrained(base, peft_path).merge_and_unload().to(device)

        # 5.2 加载测试集
        test_path = PATHS["test_file"].format(smell, smell)
        with open(test_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        texts = [item["code"] for item in data]
        true_labels_original = [item["label"] for item in data]  # 原始标签 (例如 0 和 2)
        true_labels = np.array([1 if x != 0 else 0 for x in true_labels_original])
        _, temp = codebert_predict(texts, model, tokenizer, device)
        codebert_test_probs = np.array([x[1] for x in temp])

        # ... (您加载 GCN 数据的代码保持不变) ...
        root_dir = os.path.expanduser("~/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn")
        gcn_test_input_path = os.path.join(root_dir, "graphML", smell, "test")
        gcn_test_label_path = os.path.join(root_dir, "labels", smell, "test", "labels.txt")
        gcn_test_path = os.path.join(root_dir, "processed_dataset", smell, "test")
        gcn_test_label_mapping = simple_mapping(gcn_test_input_path, gcn_test_label_path)
        gcn_model_path = f"/home/doky/project/postgraduate/model/final_model/new_gnn/{smell}/gcn_cpg_model_best.pt"

        test_dataset = CPGGraphMLDataset(
            root=gcn_test_path,
            graphml_dir=gcn_test_input_path,
            label_map={k: v for k, v in gcn_test_label_mapping.items()}
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        NODE_FEATURE_DIM = test_dataset.num_node_features
        best_model = GAT_JK_Pool(in_channels=NODE_FEATURE_DIM, hidden_channels=128, out_channels=1).to(device)
        best_model.load_state_dict(torch.load(gcn_model_path))

        _, gcn_test_probs = gcn_predict(best_model, device, test_loader)
        gcn_test_probs = np.array(gcn_test_probs)

        # CodeBERT eval
        eval_path = PATHS["eval_file"].format(smell, smell)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]
        eval_texts = [item["code"] for item in eval_data]
        eval_labels_original = [item["label"] for item in eval_data]
        eval_true_labels = np.array([1 if x != 0 else 0 for x in eval_labels_original])

        _, temp_eval = codebert_predict(eval_texts, model, tokenizer, device)
        codebert_eval_probs = np.array([x[1] for x in temp_eval])

        # GCN eval
        gcn_eval_input_path = os.path.join(root_dir, "graphML", smell, "eval")
        gcn_eval_label_path = os.path.join(root_dir, "labels", smell, "eval", "labels.txt")
        gcn_eval_path = os.path.join(root_dir, "processed_dataset", smell, "eval")
        gcn_eval_label_mapping = simple_mapping(gcn_eval_input_path, gcn_eval_label_path)

        eval_dataset = CPGGraphMLDataset(root=gcn_eval_path, graphml_dir=gcn_eval_input_path,
                                         label_map=gcn_eval_label_mapping)
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

        _, gcn_eval_probs = gcn_predict(best_model, device, eval_loader)
        gcn_eval_probs = np.array(gcn_eval_probs)

        # ==================== stacking (XGBoost) ==================
        print(f"\n--- [{smell} 开始stacking训练 (Meta-Learner: XGBoost)] ---")

        # 构建元学习器的训练集，来自Eval
        # 特征矩阵 X：[样本数, 2] 第一列是CodeBERT概率，第二列是GCN概率
        X_meta_train = np.column_stack([codebert_eval_probs, gcn_eval_probs])
        y_meta_train = eval_true_labels

        # --- 1. 计算类别权重 (替代 LR 的 class_weight='balanced') ---
        # XGBoost 使用 scale_pos_weight = count(negative) / count(positive)
        num_neg = np.sum(y_meta_train == 0)
        num_pos = np.sum(y_meta_train == 1)
        scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

        print(f"  Class Balance (Neg/Pos): {num_neg}/{num_pos}")
        print(f"  Calculated scale_pos_weight: {scale_weight:.4f}")

        # --- 2. 初始化 XGBoost 元模型 ---
        # 注意：因为只有2个输入特征，max_depth 建议设小(2-3)以防过拟合
        meta_model = XGBClassifier(
            n_estimators=100,  # 树的数量
            learning_rate=0.05,  # 学习率
            max_depth=3,  # 树的最大深度，浅树适合少特征
            scale_pos_weight=scale_weight,  # 处理样本不平衡
            eval_metric='logloss',  # 损失函数
            use_label_encoder=False,  # 消除警告
            n_jobs=-1,  # 并行加速
            random_state=42
        )

        # --- 3. 训练模型 ---
        meta_model.fit(X_meta_train, y_meta_train)

        # --- 4. 打印特征重要性 (替代 LR 的系数) ---
        # importances[0] 对应 CodeBERT, importances[1] 对应 GCN
        imps = meta_model.feature_importances_
        print(f"  Meta-Learner Feature Importances: CodeBERT={imps[0]:.4f}, GCN={imps[1]:.4f}")

        # 构建元学习器的测试集，来自test
        X_meta_test = np.column_stack([codebert_test_probs, gcn_test_probs])

        # 最终预测标签 (Hard Predictions)
        merge_preds = meta_model.predict(X_meta_test)

        # 获取最终预测概率 (用于 AUC 计算)
        merge_probs = meta_model.predict_proba(X_meta_test)[:, 1]

        # =============== 评估 ===================
        print(classification_report(true_labels, merge_preds,
                                    target_names=["none_smell", smell], digits=4))

        # --- 修改 3: 计算额外指标 (MCC, AUC, G-mean) ---

        # 1. MCC
        mcc = matthews_corrcoef(true_labels, merge_preds)

        # 2. AUC (需要概率值)
        try:
            auc = roc_auc_score(true_labels, merge_probs)
        except ValueError:
            auc = 0.0  # 处理只有一类的情况

        # 3. G-mean
        # 获取混淆矩阵元素: tn, fp, fn, tp
        cm = confusion_matrix(true_labels, merge_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity / Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
            g_mean = np.sqrt(recall * specificity)
        else:
            g_mean = 0.0

        print("-" * 15 + " Additional Metrics " + "-" * 15)
        print(f"[{smell}] MCC    : {mcc:.4f}")
        print(f"[{smell}] AUC    : {auc:.4f}")
        print(f"[{smell}] G-mean : {g_mean:.4f}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    eval()