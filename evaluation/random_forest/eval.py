import numpy as np
from sklearn.metrics import roc_auc_score

CLASS_NAMES = ["none_smell", "blob", "data_class", "feature_envy", "long_method"]
ID2LABEL = {label: i for i, label in enumerate(CLASS_NAMES)}

def evaluate(true_labels, pred_labels, model_name, code_smell_type=None, pred_probs=None):
    """
    根据真实标签、预测标签和预测概率计算评估指标。
    :true_labels: 真实标签列表 (int)
    :pred_labels: 预测标签列表 (int)
    :model_name: 模型名称
    :code_smell_type: 当前评估的异味类型 (str)
    :pred_probs: 预测为正类（当前 code_smell_type）的概率列表 (list or np.array)
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    N = len(true_labels)

    # 获取当前关注的异味类型的索引
    target_label_idx = ID2LABEL[code_smell_type] if code_smell_type else None

    # 构建二分类混淆矩阵 (One-vs-Rest)
    binary_true_labels = [] # 用于AUC计算
    
    for true_label, pred_label in zip(true_labels, pred_labels):
        # 转换为二分类逻辑：当前异味类型为正类 (1)，其他为负类 (0)
        is_target = (true_label == target_label_idx)
        is_pred_target = (pred_label == target_label_idx)
        
        binary_true_labels.append(1 if is_target else 0)

        if is_target and is_pred_target:
            TP += 1
        elif not is_target and not is_pred_target:
            TN += 1
        elif not is_target and is_pred_target:
            FP += 1
        elif is_target and not is_pred_target:
            FN += 1

    # --- 计算评估指标 ---
    
    # Accuracy
    accuracy = (TP + TN) / N if N > 0 else 0
    
    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # Recall (Sensitivity)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Specificity (用于计算 G-mean)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # G-mean = sqrt(Recall * Specificity)
    g_mean = (recall * specificity) ** 0.5
    
    # MCC
    mcc_denom = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    mcc = (TP * TN - FP * FN) / mcc_denom if mcc_denom > 0 else 0
    
    # AUC
    auc = 0.0
    if pred_probs is not None and len(set(binary_true_labels)) > 1:
        try:
            auc = roc_auc_score(binary_true_labels, pred_probs)
        except ValueError:
            auc = 0.0 # 处理只有单一类别的情况
    
    # Kappa (保留原有逻辑)
    pe = ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / N ** 2
    kappa = (accuracy - pe) / (1 - pe) if (1 - pe) > 0 else 0

    # --- 打印和保存 ---
    print(f"Model: {model_name} | Type: {code_smell_type}")
    print(f"MCC: {mcc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"G-mean: {g_mean:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print("-" * 30)

    # 路径根据您的环境需要确认存在
    output_path = f"/home/doky/project/postgraduate/eval/{model_name}/{code_smell_type}_evaluation_results.txt"
    # 为了避免路径报错，可以加个try-except或者确保文件夹存在
    try:
        with open(output_path, "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Type: {code_smell_type}\n")
            f.write(f"MCC: {mcc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"G-mean: {g_mean:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Kappa: {kappa:.4f}\n")
            f.write(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n")
            f.write(f"Total Samples: {N}\n")
            f.write(f"Confusion Matrix: \n[[{TN}, {FP}], [{FN}, {TP}]]\n")
        print("Evaluation results saved to file.")
    except FileNotFoundError:
        print(f"Error: Directory for {output_path} not found. Skipped saving file.")