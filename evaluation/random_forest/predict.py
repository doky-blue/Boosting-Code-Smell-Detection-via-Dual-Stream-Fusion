import torch
import json
import numpy as np
import joblib
import os
from eval import evaluate

# 保持与 eval.py 一致的定义
CLASS_NAMES = ["none_smell", "blob", "data_class", "feature_envy", "long_method"]
ID2LABEL = {label: i for i, label in enumerate(CLASS_NAMES)}

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义要评估的异味类型
    types = ["blob", "data_class", "feature_envy", "long_method"]

    # 加载模型和向量化器
    print("Loading model and vectorizer...")
    model = joblib.load("random_forest_code_classifier.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    
    for type_name in types:
        print(f"\nProcessing type: {type_name}...")
        predict_file = f"/home/doky/project/postgraduate/dataset/MLCQ/data/{type_name}/{type_name}.jsonl"
        
        if not os.path.exists(predict_file):
            print(f"File not found: {predict_file}")
            continue

        data = load_jsonl(predict_file)
        
        if not data:
            print(f"No data in {predict_file}")
            continue

        texts = [item["code"] for item in data]
        raw_labels = [item["label"] for item in data]
        
        # 1. 标签处理：强制使用全局 ID2LABEL 映射，保证 index 一致
        true_labels = []
        for label in raw_labels:
            if label in ID2LABEL:
                true_labels.append(ID2LABEL[label])
            else:
                # 如果遇到未知标签，默认归为 none_smell(0) 或打印警告
                # 这里假设数据标签是准确的
                true_labels.append(0) 

        # 2. 特征向量化
        codes = vectorizer.transform(texts)

        # 3. 预测 (移除原本冗余的循环，RandomForest预测是确定性的)
        predicted_labels = model.predict(codes)
        
        # 4. 获取概率 (用于计算 AUC)
        # predict_proba 返回 shape (n_samples, n_classes)
        probabilities = model.predict_proba(codes)
        
        # 获取当前目标类别的索引 (例如 blob 对应的 index)
        target_idx = ID2LABEL[type_name]
        
        # 提取目标类别的概率列
        # 注意：需要确保模型输出的概率列顺序与 CLASS_NAMES 顺序一致
        # sklearn 的 classes_ 属性通常按字典序或首次出现的顺序排序，最好确认一下
        # 这里假设模型的 classes_ 与 ID2LABEL 索引顺序一致 (0,1,2,3,4)
        if hasattr(model, "classes_"):
            # 找到 target_idx 在 model.classes_ 中的位置
            class_loc = np.where(model.classes_ == target_idx)[0]
            if len(class_loc) > 0:
                target_probs = probabilities[:, class_loc[0]]
            else:
                print(f"Warning: Class {target_idx} not found in model classes.")
                target_probs = np.zeros(len(true_labels))
        else:
            # 兜底假设
            target_probs = probabilities[:, target_idx]

        # 5. 执行评估
        evaluate(
            true_labels=true_labels, 
            pred_labels=predicted_labels, 
            model_name="random_forest", 
            code_smell_type=type_name,
            pred_probs=target_probs
        )