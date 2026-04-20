import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, 
    f1_score, 
    matthews_corrcoef, 
    roc_auc_score
)

# 1. 数据读取 (保持不变)
def load_data_from_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame(data)
    except FileNotFoundError:
        return pd.DataFrame()

# 加载数据
train_df = load_data_from_jsonl('/home/doky/project/postgraduate/dataset/MLCQ/data/train_v2.jsonl')
test_df = load_data_from_jsonl('/home/doky/project/postgraduate/dataset/MLCQ/data/test_v2.jsonl')
eval_df = load_data_from_jsonl('/home/doky/project/postgraduate/dataset/MLCQ/data/eval_v2.jsonl')

if train_df.empty:
    print("训练数据为空，退出。")
    exit()

# 2. 特征工程
print("正在进行特征向量化...")
tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))

X_train = tfidf.fit_transform(train_df['code'])
X_test = tfidf.transform(test_df['code']) if not test_df.empty else None
X_eval = tfidf.transform(eval_df['code']) if not eval_df.empty else None

y_train = train_df['label']
y_test = test_df['label'] if not test_df.empty else None
y_eval = eval_df['label'] if not eval_df.empty else None

# 3. 模型训练
print("正在训练 SVM (probability=True)...")
svm_model = SVC(
    kernel='linear', 
    class_weight='balanced', 
    decision_function_shape='ovr', 
    probability=True,  # 必须开启以计算 AUC
    random_state=42
)
svm_model.fit(X_train, y_train)
print("训练完成。\n")

# 4. 定义逐类评估函数 (Per-Class Evaluation)
def evaluate_per_class(model, X, y_true, dataset_name):
    if X is None or y_true is None:
        return

    print(f"====== {dataset_name} 分类别详细评估 ======")
    
    # 获取预测结果
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) # 形状: [样本数, 类别数]
    
    # 获取所有类别标签（例如 [0, 1, 2, 3]）
    classes = model.classes_
    
    # 用于存储结果的列表
    metrics_list = []

    # 遍历每个类别，计算 One-vs-Rest 指标
    for i, class_label in enumerate(classes):
        # --- 步骤 A: 准备二分类数据 ---
        # 将当前类别设为 1 (Positive)，其他所有类别设为 0 (Negative)
        # y_true_bin: 真实的二分类标签
        # y_pred_bin: 预测的二分类标签
        y_true_bin = (y_true == class_label).astype(int)
        y_pred_bin = (y_pred == class_label).astype(int)
        
        # 当前类别的预测概率 (用于 AUC)
        prob_current_class = y_prob[:, i]

        # --- 步骤 B: 计算基础指标 ---
        # 计算二分类混淆矩阵
        # tn: True Negative (正确排除其他类)
        # tp: True Positive (正确识别当前类)
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        
        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # --- 步骤 C: 计算高级指标 ---
        
        # 1. F1 Score (Binary)
        f1 = f1_score(y_true_bin, y_pred_bin)
        
        # 2. MCC (Matthews Correlation Coefficient)
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        
        # 3. AUC (Area Under Curve)
        try:
            auc = roc_auc_score(y_true_bin, prob_current_class)
        except ValueError:
            auc = 0.0 # 如果该类在测试集中不存在，可能报错
        
        # 4. G-mean (Geometric Mean for Binary Classification)
        # 定义为 Sensitivity 和 Specificity 的几何平均
        # 既要查全(Sens)，又要查准(Spec)
        g_mean = np.sqrt(sensitivity * specificity)

        # 存入列表
        metrics_list.append({
            "Label": class_label,
            "F1-Score": round(f1, 4),
            "MCC": round(mcc, 4),
            "AUC": round(auc, 4),
            "G-mean": round(g_mean, 4),
            "Support": np.sum(y_true_bin) # 该类在测试集中的样本数
        })

    # --- 步骤 D: 输出表格 ---
    # 使用 Pandas 展示漂亮的表格
    results_df = pd.DataFrame(metrics_list)
    
    # 打印表格
    print(results_df.to_string(index=False))
    print("-" * 60)

# 5. 执行评估
evaluate_per_class(svm_model, X_test, y_test, "测试集 (Test Set)")
evaluate_per_class(svm_model, X_eval, y_eval, "验证集 (Eval Set)")