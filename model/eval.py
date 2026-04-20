import os

CLASS_NAMES = ["none_smell", "blob", "data_class", "feature_envy", "long_method"]
ID2LABEL = {name: idx for idx, name in enumerate(CLASS_NAMES)}

def evaluate(y_true, y_pred, model_name, code_smell_type=None):
    """
    二分类评估：正类=当前 smell，负类=none_smell
    """
    assert code_smell_type is not None
    pos_label = ID2LABEL[code_smell_type]          # 1
    neg_label = ID2LABEL["none_smell"]             # 0

    TP = TN = FP = FN = 0
    for t, p in zip(y_true, y_pred):
        if t == pos_label and p == pos_label: TP += 1
        elif t == neg_label and p == neg_label: TN += 1
        elif t == neg_label and p == pos_label: FP += 1
        elif t == pos_label and p == neg_label: FN += 1

    N = len(y_true)
    acc = (TP + TN) / N
    prec = TP / (TP + FP) if (TP + FP) else 0
    rec = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    mcc_num = TP * TN - FP * FN
    mcc_den = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    mcc = mcc_num / mcc_den if mcc_den else 0
    print(f"Model: {model_name}  Smell: {code_smell_type}")
    print(f"Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}  MCC: {mcc:.4f}")
    print(f"TP:{TP}  TN:{TN}  FP:{FP}  FN:{FN}")

    # 保存
    out_dir = f"/home/doky/project/postgraduate/eval/{model_name}/"
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{code_smell_type}_evaluation_results.txt", "w") as f:
        f.write(f"{model_name}  {code_smell_type}\nAcc {acc:.4f}  Prec {prec:.4f}  Rec {rec:.4f}  F1 {f1:.4f}  MCC {mcc:.4f}\n")