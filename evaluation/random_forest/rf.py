import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # <--- 关键修改：导入随机森林
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

# 2. 特征工程 (保持不变)
print("正在进行特征向量化...")
# Random Forest 对高维稀疏数据也能处理，但在文本分类中，线性 SVM 往往略优于 RF
tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))

X_train = tfidf.fit_transform(train_df['code'])
X_test = tfidf.transform(test_df['code']) if not test_df.empty else None
X_eval = tfidf.transform(eval_df['code']) if not eval_df.empty else None

y_train = train_df['label']
y_test = test_df['label'] if not test_df.empty else None
y_eval = eval_df['label'] if not eval_df.empty else None

# 3. 模型训练 (Random Forest)
print("正在训练 Random Forest 模型...")

# --- 关键修改开始 ---
rf_model = RandomForestClassifier(
    n_estimators=200,          # 树的数量，越多通常越稳定，但计算越慢
    class_weight='balanced',   # 处理样本不平衡：根据频率自动调整权重
    n_jobs=-1,                 # 使用所有 CPU 核心并行训练 (加速!)
    random_state=42,           # 保证结果可复现
    max_depth=None             # 树的深度不限，直到分不开为止 (也可设为 20-50 防止过拟合)
)
# --- 关键修改结束 ---

rf_model.fit(X_train, y_train)
print("训练完成。\n")

# 4. 定义逐类评估函数 (逻辑完全复用)
def evaluate_per_class(model, X, y_true, dataset_name):
    if X is None or y_true is None:
        return

    print(f"====== {dataset_name} 分类别详细评估 (Random Forest) ======")
    
    # 获取预测结果
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) # RF 天生支持这个，不需要 probability=True
    
    classes = model.classes_
    metrics_list = []

    for i, class_label in enumerate(classes):
        # 构造二分类标签 (One-vs-Rest)
        y_true_bin = (y_true == class_label).astype(int)
        y_pred_bin = (y_pred == class_label).astype(int)
        
        # 提取当前类的概率
        # 注意：如果某类在训练集中存在但在测试集完全没出现，需要做边界检查，
        # 但这里的 logic 是基于 model.classes_ 遍历的，通常是安全的。
        if y_prob.shape[1] > i:
            prob_current_class = y_prob[:, i]
        else:
            prob_current_class = np.zeros_like(y_true_bin, dtype=float)

        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 计算指标
        f1 = f1_score(y_true_bin, y_pred_bin)
        mcc = matthews_corrcoef(y_true_bin, y_pred_bin)
        
        try:
            auc = roc_auc_score(y_true_bin, prob_current_class)
        except ValueError:
            auc = 0.0 
        
        g_mean = np.sqrt(sensitivity * specificity)

        metrics_list.append({
            "Label": class_label,
            "F1-Score": round(f1, 4),
            "MCC": round(mcc, 4),
            "AUC": round(auc, 4),
            "G-mean": round(g_mean, 4),
            "Support": np.sum(y_true_bin)
        })

    results_df = pd.DataFrame(metrics_list)
    print(results_df.to_string(index=False))
    print("-" * 60)

# 5. 执行评估
evaluate_per_class(rf_model, X_test, y_test, "测试集 (Test Set)")
evaluate_per_class(rf_model, X_eval, y_eval, "验证集 (Eval Set)")