import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    recall_score
)


# 1. 数据读取函数 (保持不变)
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
        print(f"错误: 找不到文件 {file_path}")
        return pd.DataFrame()


# --- 主流程 ---

# 加载数据
train_df = load_data_from_jsonl('/home/doky/project/postgraduate/dataset/MLCQ/data/train_v2.jsonl')
test_df = load_data_from_jsonl('/home/doky/project/postgraduate/dataset/MLCQ/data/test_v2.jsonl')
eval_df = load_data_from_jsonl('/home/doky/project/postgraduate/dataset/MLCQ/data/eval_v2.jsonl')

if train_df.empty:
    print("训练数据为空，退出。")
    exit()

# 2. 特征工程 (TF-IDF)
print("\n正在进行特征向量化...")
tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))

# 仅在训练集上拟合
X_train = tfidf.fit_transform(train_df['code'])
# 测试和验证集仅转换
X_test = tfidf.transform(test_df['code']) if not test_df.empty else None
X_eval = tfidf.transform(eval_df['code']) if not eval_df.empty else None

y_train = train_df['label']
y_test = test_df['label'] if not test_df.empty else None
y_eval = eval_df['label'] if not eval_df.empty else None

# 3. 模型训练
print("\n正在训练 SVM 模型 (开启概率预测，可能稍慢)...")

# 【关键修改】 probability=True
# 必须开启这个选项才能计算 AUC，否则 model.predict_proba() 会报错
svm_model = SVC(
    kernel='linear',
    class_weight='balanced',
    decision_function_shape='ovr',
    probability=True,  # <--- 为了计算 AUC
    random_state=42
)

svm_model.fit(X_train, y_train)
print("模型训练完成。")


# 4. 定义综合评估函数
def evaluate_comprehensive(model, X, y, dataset_name):
    if X is None or y is None:
        return

    print(f"\n====== {dataset_name} 详细评估报表 ======")

    # 预测类别 (用于 F1, MCC, G-mean)
    y_pred = model.predict(X)

    # 预测概率 (用于 AUC)
    # 返回的是一个矩阵，每一行对应每个类别的概率
    y_prob = model.predict_proba(X)

    # --- 指标计算 ---

    # 1. F1 Score (Weighted: 考虑样本不平衡; Macro: 平均看待每个类)
    f1 = f1_score(y, y_pred, average='weighted')

    # 2. MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(y, y_pred)

    # 3. AUC (Area Under Curve) - 多分类需指定策略
    # multi_class='ovr': One-vs-Rest 策略
    try:
        auc = roc_auc_score(y, y_prob, multi_class='ovr', average='weighted')
    except ValueError:
        # 如果测试集中缺少某些类别，AUC 计算可能会报错
        auc = 0.0
        print("警告：测试集类别不全，无法准确计算 AUC")

    # 4. G-mean (Geometric Mean)
    # 定义：所有类别 Recall(召回率) 的几何平均数
    # G-mean = (Recall_1 * Recall_2 * ... * Recall_k)^(1/k)
    # 如果某类 Recall 为 0，G-mean 即为 0，这代表模型对某一类完全失效
    recalls = recall_score(y, y_pred, average=None)  # 获取每个类的 Recall
    g_mean = np.prod(recalls) ** (1 / len(recalls))

    # --- 打印结果 ---
    print(f"{'指标 (Metric)':<20} | {'数值 (Value)':<10}")
    print("-" * 35)
    print(f"{'F1 Score (Weighted)':<20} | {f1:.4f}")
    print(f"{'MCC':<20} | {mcc:.4f}")
    print(f"{'AUC (OVR, Weighted)':<20} | {auc:.4f}")
    print(f"{'G-mean':<20} | {g_mean:.4f}")
    print("-" * 35)

    # 附带打印每个类的 Recall，方便分析 G-mean 低的原因
    print(f"各类别 Recall分布: {np.round(recalls, 4)}")


# 5. 执行评估
evaluate_comprehensive(svm_model, X_test, y_test, "测试集 (Test Set)")
evaluate_comprehensive(svm_model, X_eval, y_eval, "验证集 (Eval Set)")