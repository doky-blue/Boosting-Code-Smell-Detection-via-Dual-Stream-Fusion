import json
import torch
import os
import numpy as np
from torch.functional import F
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from peft import PeftModel
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
        batch = texts[i: i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits          # [B, 2]
        logit_probs = F.softmax(logits, dim=1)  # 形状仍为 [B, 2]
        preds.append(logits.argmax(1).cpu())
        all_logits.append(logit_probs.cpu())
    return torch.cat(preds).numpy(), torch.cat(all_logits).numpy()


def gcn_predict(model, device, loader, threshold=0.70):
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze(1)

            predicted_probs = torch.sigmoid(out)  # 形状 (B,)
            preds = (predicted_probs > threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(predicted_probs.cpu().numpy())

    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    return y_pred, y_probs

def merge_res(codebert_preds, gcn_preds):
    res_preds = []
    for i,j in zip(codebert_preds, gcn_preds):
        if i == 0 and j == 0:
            res_preds.append(0)
        else:
            res_preds.append(1)
    return res_preds

def weight_merge(codebert_preds, gcn_preds, weight=0.6, threshold=0.40):
    res_preds = []
    
    # 这里用于生成 0/1 标签
    combined_probs = (codebert_preds * weight) + (gcn_preds * (1 - weight))
    
    for prob in combined_probs:
        if prob > threshold:
            res_preds.append(1)
        else:
            res_preds.append(0)

    return res_preds

PATHS = {
    "base_model": "/home/doky/Llama/codebert-base/",
    "peft_root" : "/home/doky/project/postgraduate/model/final_model/codebert/{}/",
    "test_file" : "/home/doky/project/postgraduate/dataset/MLCQ/data/{}/{}_test_four_model.jsonl",
    "eval_file" : "/home/doky/project/postgraduate/dataset/MLCQ/data/{}/{}_eval_four_model.jsonl",
    "save_dir"  : "/home/doky/project/postgraduate/eval/four_model_fix/",
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
        true_labels_original = [item["label"] for item in data]

        # 二值化标签
        true_labels = np.array([1 if x != 0 else 0 for x in true_labels_original])

        # 5.3 预测
        code_bert_pred_labels, temp = codebert_predict(texts, model, tokenizer, device)
        codebert_logits = []
        for i in range(0, len(temp)):
            codebert_logits.append(temp[i][1])  # 获取 positive 类的概率

        # GCN 预测部分
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

        gcn_pred_labels, gcn_logits_preds = gcn_predict(best_model, device, test_loader)

        # 确保为 numpy 数组
        gcn_logits_preds = np.array(gcn_logits_preds)
        codebert_logits = np.array(codebert_logits)

        # 网格搜索参数
        weights_to_try = np.linspace(0, 1, 21)
        threshold_to_try = np.linspace(0.01, 0.99, 99)

        # 加载验证集
        eval_path = PATHS["eval_file"].format(smell, smell)
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]

        eval_texts = [item["code"] for item in eval_data]
        codebert_eval_true_labels_original = [item["label"] for item in eval_data]
        codebert_eval_true_labels = np.array([1 if x != 0 else 0 for x in codebert_eval_true_labels_original])

        gcn_eval_input_path = os.path.join(root_dir, "graphML", smell, "eval")
        gcn_eval_label_path = os.path.join(root_dir, "labels", smell, "eval", "labels.txt")
        gcn_eval_path = os.path.join(root_dir, "processed_dataset", smell, "eval")

        gcn_eval_label_mapping = simple_mapping(gcn_eval_input_path, gcn_eval_label_path)

        eval_dataset = CPGGraphMLDataset(
            root=gcn_eval_path,
            graphml_dir=gcn_eval_input_path,
            label_map={k: v for k, v in gcn_eval_label_mapping.items()}
        )
        eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

        gcn_eval_pred_labels, gcn_eval_logits_preds = gcn_predict(best_model, device, eval_loader)

        codebert_eval_pred_labels, temp = codebert_predict(eval_texts, model, tokenizer, device)
        codebert_eval_logits = []
        for i in range(0, len(temp)):
            codebert_eval_logits.append(temp[i][1])

        gcn_eval_logits_preds = np.array(gcn_eval_logits_preds)
        codebert_eval_logits = np.array(codebert_eval_logits)


        best_f1 = -1.0
        best_weight = 0.0
        best_threshold = 0.0

        print(f"\n--- [{smell}] 开始网格搜索... ---")
        
        for weight in weights_to_try:
            combined_probs = (codebert_eval_logits * weight) + (gcn_eval_logits_preds * (1 - weight))
            for threshold in threshold_to_try:
                current_preds = (combined_probs > threshold).astype(int)
                current_f1 = f1_score(codebert_eval_true_labels, current_preds, pos_label=1, zero_division=0)
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_weight = weight
                    best_threshold = threshold

        print(f"--- [{smell}] 网格搜索完成 ---")
        print(f"最佳 (Smell 类) F1-score: {best_f1:.4f}")
        print(f"最佳 Weight: {best_weight:.2f}")
        print(f"最佳 Threshold: {best_threshold:.2f}")

        # 4. 使用找到的最佳参数生成最终预测
        # 生成最终的硬标签预测
        merge_logit_preds = weight_merge(codebert_logits, gcn_logits_preds,
                                         weight=best_weight, threshold=best_threshold)
        
        # --- 修改 2: 生成最终的概率预测 (用于 AUC 计算) ---
        final_test_probs = (codebert_logits * best_weight) + (gcn_logits_preds * (1 - best_weight))

        # 5.4 基础评估报告
        print(classification_report(true_labels, merge_logit_preds,
                                    target_names=["none_smell", smell], digits=4))

        # --- 修改 3: 计算额外指标 (MCC, AUC, G-mean) ---
        
        # 1. MCC
        mcc = matthews_corrcoef(true_labels, merge_logit_preds)
        
        # 2. AUC (Area Under Curve) - 需要概率值
        try:
            auc = roc_auc_score(true_labels, final_test_probs)
        except ValueError:
            auc = 0.0 # 处理只有一个类别的情况
            
        # 3. G-mean (Geometric Mean) = sqrt(Sensitivity * Specificity)
        # 获取混淆矩阵元素: tn, fp, fn, tp
        cm = confusion_matrix(true_labels, merge_logit_preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0      # Sensitivity / Recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # Specificity
            g_mean = np.sqrt(recall * specificity)
        else:
            # 极端情况处理
            g_mean = 0.0

        print("-" * 15 + " Additional Metrics " + "-" * 15)
        print(f"[{smell}] MCC    : {mcc:.4f}")
        print(f"[{smell}] AUC    : {auc:.4f}")
        print(f"[{smell}] G-mean : {g_mean:.4f}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    eval()