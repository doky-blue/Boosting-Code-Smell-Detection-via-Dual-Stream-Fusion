import os, json, torch, numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from peft import PeftModel
from sklearn.metrics import classification_report
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. 路径配置 —— 只需改这里
# ------------------------------------------------------------------
PATHS = {
    "base_model": "/home/doky/Llama/codebert-base/",
    "peft_root" : "/home/doky/project/postgraduate/model/four_model/output/{}/2025-09-12/",
    "test_file" : "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/test.jsonl",
    "save_dir"  : "/home/doky/project/postgraduate/eval/four_model_fix/",
}
os.makedirs(PATHS["save_dir"], exist_ok=True)

# ------------------------------------------------------------------
# 2. 标签映射（二分类：none_smell=0, 当前 smell=1）
# ------------------------------------------------------------------
# LABEL_MAP = {"none_smell": 0}          # 统一把 none_smell 映射到 0
# for smell in ["blob", "data_class", "feature_envy", "long_method"]:
#     LABEL_MAP[smell] = 1               # 四个 smell 全部映射到 1

# ------------------------------------------------------------------
# 3. 加载数据
# ------------------------------------------------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# ------------------------------------------------------------------
# 4. 预测函数（单次，无投票）
# ------------------------------------------------------------------
@torch.no_grad()
def predict(texts, model, tokenizer, device, batch_size=16):
    model.eval()
    # preds = []
    # all_probs = []
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits          # [B, 2]

        logit_diff = logits[:, 1] - logits[:, 0]
        all_logits.append(logit_diff.cpu())

    return torch.cat(all_logits)


    #     probs = torch.softmax(logits, dim=1)  # 在类别维度(dim=1)上应用softmax
    #     preds.append(probs.cpu())
    #     all_probs.append(probs.cpu())
    #
    # # 将所有批次的结果拼接成一个张量，并转换为 numpy 数组
    # return torch.cat(all_probs).numpy()


def predict_multiclass_with_softmax(texts, models, tokenizer, device):
    """
    使用概率归一化（Softmax）方法进行五分类预测。

    Args:
        texts (list of str): 输入的文本列表。
        models (list): 包含四个二分类模型的列表 [model_01, model_02, model_03, model_04]。
        tokenizer: 分词器。
        device: 'cpu' 或 'cuda'。

    Returns:
        dict: 包含最终预测类别和每个类别的概率。
    """
    # 步骤 1: 获取类别 1, 2, 3, 4 的 logits
    logit_s1 = predict(texts, models[0], tokenizer, device)
    logit_s2 = predict(texts, models[1], tokenizer, device)
    logit_s3 = predict(texts, models[2], tokenizer, device)
    logit_s4 = predict(texts, models[3], tokenizer, device)

    # 步骤 2: 构建分数矩阵
    # 为类别 0 创建一个全为 0 的 logit 向量
    num_texts = len(texts)
    logit_s0 = torch.zeros(num_texts)

    # 将所有 logits 堆叠成一个矩阵，形状为 [num_texts, 5]
    # 每一行代表一个文本的 [s0, s1, s2, s3, s4] 分数
    score_matrix = torch.stack([logit_s0, logit_s1, logit_s2, logit_s3, logit_s4], dim=1)

    # 步骤 3: 应用 Softmax 函数
    # F.softmax 会在指定的维度（dim=1，即类别维度）上计算概率
    probabilities = F.softmax(score_matrix, dim=1)

    # 步骤 4: 做出最终预测
    # torch.argmax 在类别维度上找到最大概率的索引，这个索引就是预测的类别
    predictions = torch.argmax(probabilities, dim=1)

    return {
        "predictions": predictions.numpy(),
        "probabilities": probabilities.numpy()
    }

# ------------------------------------------------------------------
# 5. 主流程
# ------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(PATHS["base_model"])

    pred_labels = []
    true_pred_label = []

    for smell in ["blob", "data_class", "feature_envy", "long_method"]:
        print(f"\n========== {smell} ==========")
        # 5.1 加载模型
        base = RobertaForSequenceClassification.from_pretrained(
            PATHS["base_model"], num_labels=2)
        peft_path = PATHS["peft_root"].format(smell)
        model = PeftModel.from_pretrained(base, peft_path).merge_and_unload().to(device)

        # 5.2 加载测试集
        test_path = PATHS["test_file"].format(smell, smell)
        data = load_jsonl(test_path)
        texts = [item["code"] for item in data]
        true_labels = [item["label"] for item in data]  # 保持原数字 0 或 pos_label

        # 5.3 预测
        pred_labels.append(predict(texts, model, tokenizer, device))

    for i in range(0, 1335):
        temp_smell_logits = [0, 0, 0, 0, 0]
        for j in range(0, 4):
            if pred_labels[j][i][1] > 0.5:
                temp_smell_logits[j + 1] = pred_labels[j][i][1]

        if max(temp_smell_logits) == 0:
            true_pred_label.append(0)
        else:
            true_pred_label.append(temp_smell_logits.index(max(temp_smell_logits)))

    # 5.4 评估
    print(classification_report(true_labels, true_pred_label,
                                target_names=["none_smell", "blob", "data_class", "feature_envy", "long_method"], digits=4))

    out_dir = f"/home/doky/project/postgraduate/eval/four_model_fix/"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{smell}_evaluation_results.txt", "w") as f:
        f.write(classification_report(true_labels, pred_labels, target_names=["none_smell", smell], digits=4))

    # 5.5 保存预测结果（可选）
    save_path = os.path.join(PATHS["save_dir"], f"{smell}_preds.jsonl")
    with open(save_path, "w") as f:
        for t, p in zip(true_labels, pred_labels):
            f.write(json.dumps({"true": int(t), "pred": int(p)}) + "\n")
    print(f"Predictions saved -> {save_path}")

def softmax_logits():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained(PATHS["base_model"])
    base_model = RobertaForSequenceClassification.from_pretrained(
            PATHS["base_model"], num_labels=2)

    models = []
    smells = ["blob", "data_class", "feature_envy", "long_method"]
    for smell in smells:
        peft_path = PATHS["peft_root"].format(smell)
        model = PeftModel.from_pretrained(base_model, peft_path).merge_and_unload().to(device)
        models.append(model)

    # 5.2 加载测试集
    test_path = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/test.jsonl"
    data = load_jsonl(test_path)
    texts = [item["code"] for item in data]
    true_labels = [item["label"] for item in data]  # 保持原数字 0 或 pos_label

    result = predict_multiclass_with_softmax(texts, models, tokenizer, device)

    print("预测类别:", result["predictions"])
    print("每个类别的概率:\n", result["probabilities"])

# --- 使用示例 ---
# models = [model_01, model_02, model_03, model_04]
# test_texts = ["这是一个示例文本。", "这是另一个。"]
# result = predict_multiclass_with_softmax(test_texts, models, tokenizer, device)

if __name__ == "__main__":
    # main()
    softmax_logits()