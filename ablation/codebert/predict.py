import torch
import json
import numpy as np

from eval import evaluate

from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from peft import PeftModel

CLASS_NAMES = ["none_smell", "blob", "data_class", "feature_envy", "long_method"]
ID2LABEL = {label: i for i, label in enumerate(CLASS_NAMES)}
print(ID2LABEL)

MODEL_NAMES = ["code_llm_with_fine_tuning", "code_llm_without_fine_tuning", "astnn", "llm_api", "fine_tuning_llm_and_gnn", "random_forest"]

def compute_metrics(predictions, labels):
    """
    计算模型预测的准确率、F1 分数、精确率和召回率
    Args:
        predictions (np.ndarray): 模型预测的标签
        labels (np.ndarray): 实际标签
    Returns:
        dict: 包含准确率、F1 分数、精确率和召回率的字典
    """
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# 2. 加载数据
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def predict_with_logits(texts, model, tokenizer, batch_size=8):
    logits_list = []
    predictions = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # 将输入数据移动到与模型相同的设备上
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()  # 注意这里改为.cpu()
        
        logits_list.append(logits.cpu().numpy())  # 将logits移回CPU再转换为numpy
        predictions.extend(preds)
    
    # 合并所有batch的logits
    all_logits = np.concatenate(logits_list, axis=0)
    # print(len(predictions))
    return predictions, all_logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    types = ["blob", "data_class", "feature_envy", "long_method"]

    # 1. 加载模型和tokenizer
    model_name = "/home/doky/Llama/codebert-base/"
    peft_model_id = "/home/doky/project/postgraduate/model/code_llm_with_fine_tuning/output/"

    base_model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CLASS_NAMES)
        )
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
    model = peft_model.merge_and_unload()
    model = model.to(device)
    model.eval()

    for type in types:
        predict_file = f"/home/doky/project/postgraduate/dataset/MLCQ/data/{type}/{type}.jsonl"
        data = load_jsonl(predict_file)
        texts = [item["code"] for item in data]
        true_labels = [item["label"] for item in data]
        # print(true_labels)

        # 4. 执行预测和评估
        predictions = []
        predicted_labels = []
        error = 0
        for i in range(3):
            temp, _ = predict_with_logits(texts, model, tokenizer)
            predictions.append(temp)
        for i in range(len(predictions[0])):
            j, k, l = predictions[0][i], predictions[1][i], predictions[2][i]
            if j == k:
                predicted_labels.append(j)
            elif j == l:
                predicted_labels.append(j)
            elif k == l:
                predicted_labels.append(k)
            else:
                error += 1
                predicted_labels.append(9)
        print(error)

        # 处理标签（如果是字符串转换为数字）
        if isinstance(true_labels[0], str):
            label_map = {label: idx for idx, label in enumerate(set(true_labels))}
            true_labels = [label_map[label] for label in true_labels]
        
        # evaluate(true_labels, predicted_labels, "code_llm_with_fine_tuning", code_smell_type=type)

        # # 计算指标
        # print(predicted_labels)
        # accuracy = accuracy_score(true_labels, predicted_labels)
        # precision = precision_score(true_labels, predicted_labels, average='weighted')
        # recall = recall_score(true_labels, predicted_labels, average='weighted')
        # f1 = f1_score(true_labels, predicted_labels, average='weighted')
        # class_report = classification_report(true_labels, predicted_labels)
        # conf_matrix = confusion_matrix(true_labels, predicted_labels)

        # # 打印结果
        # print(f"Accuracy: {accuracy:.4f}")
        # print(f"Weighted Precision: {precision:.4f}")
        # print(f"Weighted Recall: {recall:.4f}")
        # print(f"Weighted F1 Score: {f1:.4f}")
        # print("\nClassification Report:")
        # print(class_report)
        # print("\nConfusion Matrix:")
        # print(conf_matrix)

        # # 5. 保存结果
        # eval_results = {
        #     "accuracy": accuracy,
        #     "weighted_precision": precision,
        #     "weighted_recall": recall,
        #     "weighted_f1": f1,
        #     "classification_report": class_report,
        #     "confusion_matrix": conf_matrix.tolist(),
        # }

        # with open(f"{type}_evaluation_results.json", "w") as f:
        #     json.dump(eval_results, f, indent=2)