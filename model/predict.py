import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    matthews_corrcoef,
    f1_score
)

# ------------------------------------------------------------------
# 1. 路径配置
# ------------------------------------------------------------------
PATHS = {
    "base_model": "/home/doky/Llama/codebert-base/",
    "peft_root": "/home/doky/project/postgraduate/model/four_model/output/{}/2025-12-04/",
    # "peft_root": "/home/doky/project/postgraduate/model/final_model/codebert/{}/",
    "test_file": "/home/doky/project/postgraduate/dataset/MLCQ/data/{}/{}_test_four_model.jsonl",
    "save_dir": "/home/doky/project/postgraduate/eval/four_model_fix/",
}
os.makedirs(PATHS["save_dir"], exist_ok=True)

# ------------------------------------------------------------------
# 2. 标签映射
# ------------------------------------------------------------------
LABEL_MAP = {"none_smell": 0}
for smell in ["blob", "data_class", "feature_envy", "long_method"]:
    LABEL_MAP[smell] = 1


# ------------------------------------------------------------------
# 3. 加载数据
# ------------------------------------------------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ------------------------------------------------------------------
# 4. 预测函数 (返回概率值)
# ------------------------------------------------------------------
@torch.no_grad()
def predict(texts, model, tokenizer, device, batch_size=16):
    model.eval()
    probs_list = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits  # [B, 2]

        # 使用 Softmax 将 logits 转换为概率
        # dim=1 表示在类别维度进行 softmax
        probs = torch.softmax(logits, dim=1)

        # 取出正类 (label=1, 即有 smell) 的概率
        probs_list.append(probs[:, 1].cpu())

    # 返回 numpy 数组格式的概率，形状为 [N, ]
    return torch.cat(probs_list).numpy()


# ------------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载 Tokenizer (只需加载一次)
    tokenizer = RobertaTokenizer.from_pretrained(PATHS["base_model"])

    for smell in ["blob", "data_class", "feature_envy", "long_method"]:
        print(f"\n========== {smell} ==========")

        # 5.1 加载基础模型和 LoRA 权重
        base = RobertaForSequenceClassification.from_pretrained(
            PATHS["base_model"], num_labels=2)
        peft_path = PATHS["peft_root"].format(smell)

        try:
            model = PeftModel.from_pretrained(base, peft_path).merge_and_unload().to(device)
        except Exception as e:
            print(f"Error loading model for {smell}: {e}")
            continue

        # 5.2 加载测试集
        test_path = PATHS["test_file"].format(smell, smell)
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}")
            continue

        data = load_jsonl(test_path)
        texts = [item["code"] for item in data]
        true_labels = [int(item["label"]) for item in data]

        # 5.3 模型预测
        print("Predicting...")
        pred_probs = predict(texts, model, tokenizer, device)

        # 将概率转换为 0/1 标签 (阈值 0.5)
        pred_labels = (pred_probs >= 0.5).astype(int)

        # 5.4 计算各项指标
        # (1) AUC
        auc_score = roc_auc_score(true_labels, pred_probs)

        # (2) MCC (Matthews Correlation Coefficient)
        mcc_score = matthews_corrcoef(true_labels, pred_labels)

        # (3) G-mean
        # 混淆矩阵: tn (真负), fp (假正), fn (假负), tp (真正)
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        g_mean = np.sqrt(sensitivity * specificity)

        # (4) F1-score (宏平均或针对正类的F1，这里使用 classification_report 展示详细信息)
        # 也可以单独计算: f1 = f1_score(true_labels, pred_labels)

        # 生成详细分类报告 (包含 Precision, Recall, F1)
        cls_report = classification_report(true_labels, pred_labels,
                                           target_names=["none_smell", smell], digits=4)

        # 打印结果到控制台
        print(f"AUC    : {auc_score:.4f}")
        print(f"MCC    : {mcc_score:.4f}")
        print(f"G-mean : {g_mean:.4f}")
        print(cls_report)

        # 5.5 绘制并保存 ROC 曲线
        fpr, tpr, thresholds = roc_curve(true_labels, pred_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {smell}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        # 保存 ROC 图片
        roc_save_path = os.path.join(PATHS["save_dir"], f"{smell}_roc_curve.png")
        plt.savefig(roc_save_path)
        plt.close()  # 关闭画布，避免内存泄漏
        print(f"ROC curve saved -> {roc_save_path}")

        # 5.6 保存所有评估结果到文本文件
        eval_txt_path = os.path.join(PATHS["save_dir"], f"{smell}_evaluation_results.txt")
        with open(eval_txt_path, "w") as f:
            f.write(f"Test File: {test_path}\n")
            f.write("-" * 30 + "\n")
            f.write(f"AUC     : {auc_score:.4f}\n")
            f.write(f"MCC     : {mcc_score:.4f}\n")
            f.write(f"G-mean  : {g_mean:.4f}\n")
            f.write("-" * 30 + "\n")
            f.write("Classification Report:\n")
            f.write(cls_report)

        # 5.7 保存详细预测数据 (True Label, Pred Label, Probability)
        # 这对于后续错误分析非常有用
        preds_jsonl_path = os.path.join(PATHS["save_dir"], f"{smell}_preds.jsonl")
        with open(preds_jsonl_path, "w") as f:
            for t, p, prob in zip(true_labels, pred_labels, pred_probs):
                f.write(json.dumps({
                    "true": int(t),
                    "pred": int(p),
                    "prob": float(prob)
                }) + "\n")
        print(f"Predictions saved -> {preds_jsonl_path}")


if __name__ == "__main__":
    main()