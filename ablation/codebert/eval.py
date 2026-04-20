CLASS_NAMES = ["none_smell", "blob", "data_class", "feature_envy", "long_method"]
ID2LABEL = {label: i for i, label in enumerate(CLASS_NAMES)}

def evaluate(true_labels, pred_labels, model_name, code_smell_type=None):
    """
    根据真实标签和预测标签计算评估指标，并打印分类报告和混淆矩阵。
    :true_labels: 真实标签列表
    :pred_labels: 预测标签列表
    :model_name: 模型名称
    评估指标包括 Acc Precision Recall F1 MCC Kappa Confusion_Matrix  
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    N = len(true_labels)

    label = ID2LABEL[code_smell_type] if code_smell_type else None

    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label == pred_label:
            if true_label == label:
                TP += 1
            else:
                TN += 1
        else:
            if true_label == label:
                FN += 1
            else:
                FP += 1
    # 计算评估指标
    accuracy = (TP + TN) / N
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mcc = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5 if (
            (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) > 0 else 0
    kappa = (accuracy - ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / N ** 2) / (
            1 - ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / N ** 2) if (
            1 - ((TP + FP) * (TP + FN) + (TN + FP) * (TN + FN)) / N ** 2) > 0 else 0
    # 打印评估指标
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f"Total Samples: {N}")
    print(f"Confusion Matrix: \n[[{TN}, {FP}], [{FN}, {TP}]]")

    with open(f"/home/doky/project/postgraduate/eval/{model_name}/{code_smell_type}_evaluation_results.txt", "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"MCC: {mcc:.4f}\n")
        f.write(f"Kappa: {kappa:.4f}\n")
        f.write(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}\n")
        f.write(f"Total Samples: {N}\n")
        f.write(f"Confusion Matrix: \n[[{TN}, {FP}], [{FN}, {TP}]]\n")
    print("Evaluation results saved to file.")