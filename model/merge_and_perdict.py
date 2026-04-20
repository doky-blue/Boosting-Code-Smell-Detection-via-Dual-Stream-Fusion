import torch
import json
import random
from peft import PeftModel
from sklearn.metrics import classification_report
from transformers import RobertaForSequenceClassification, RobertaTokenizer

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

@torch.no_grad()
def predict(texts, model, tokenizer, device, batch_size=16):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        logits = model(**enc).logits          # [B, 2]
        preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).numpy()

if __name__ == "__main__":
    # 基础配置
    base_model_path = "/home/doky/Llama/codebert-base/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained(base_model_path)
    base_model = RobertaForSequenceClassification.from_pretrained(base_model_path, num_labels=2)

    smell_list = ["blob", "data_class", "feature_envy", "long_method"]

    # test_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/test.jsonl"
    test_file = "/home/doky/project/postgraduate/dataset/MLCQ/multi_label/test.jsonl"

    with open(test_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    codes = [item["code"] for item in data]
    true_labels = [item["labels"] for item in data]
    pred_labels = []
    true_pred_labels = []
    for _ in range(len(true_labels)):
        true_pred_labels.append([0, 0, 0, 0, 0])

    # 实际异味标签
    label = 1

    # blob_pred_labels = []
    # data_class_pred_labels = []
    # feature_envy_pred_labels = []
    # long_method_pred_labels = []
    # all_preds = [blob_pred_labels, data_class_pred_labels, feature_envy_pred_labels, long_method_pred_labels]

    for smell in smell_list:
        peft_path = f"/home/doky/project/postgraduate/model/four_model/output/{smell}/2025-09-25/"

        peft_model = PeftModel.from_pretrained(base_model, peft_path).merge_and_unload().to(device)

        model_pred_labels = predict(codes, peft_model, tokenizer, device)
        for i in range(0, len(model_pred_labels)):
            # if i == 1:
            #     all_preds[label].append(label)
            # if i == 0:
            #     all_preds[label].append(0)
            if model_pred_labels[i] == 1:
                true_pred_labels[i][label] = 1
                true_pred_labels[i][0] = 0
            elif true_pred_labels[i][0] == 0 and true_pred_labels[i][1] != 1 and true_pred_labels[i][2] != 1 and true_pred_labels[i][3] != 1 and true_pred_labels[i][4] != 1:
                true_pred_labels[i][0] = 1
        label += 1

    print(classification_report(true_labels, true_pred_labels, digits=4))
