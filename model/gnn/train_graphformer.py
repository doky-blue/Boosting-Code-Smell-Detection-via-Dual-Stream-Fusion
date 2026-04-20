# file: train_graphformer.py

import os
import datetime
import torch
import json
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, recall_score
from dataset import JavaGraphDataset
from model_graphormer import GraphormerClassifier

def collate_fn(batch_list):
    keys = batch_list[0].keys()
    batch = {}
    for k in keys:
        values = [getattr(sample, k) for sample in batch_list]
        try:
            if k == 'num_nodes' or k == 'max_nodes':
                batch[k] = torch.stack(values, dim=0)
                continue
            batch[k] = torch.stack(values, dim=0)
        except RuntimeError as e:
            print(f"[ERROR] Key: {k}")
            for i, v in enumerate(values):
                print(f"  Sample {i}: {v.shape}")
            raise e
    return batch


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    y_true, y_prob = [], []
    for data in loader:
        data = {k: v.to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}
        logits = model(data)
        probs = torch.sigmoid(logits)
        y_true.append(data["y"].cpu())
        y_prob.append(probs.cpu())

    # 形状将是 [N, 1]，我们使用 .ravel() 将其变为 [N] 以便 sklearn 处理
    y_true = torch.cat(y_true).numpy().ravel()
    y_prob = torch.cat(y_prob).numpy().ravel()

    # 寻找最佳阈值
    ths = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y_true, y_prob >= t, zero_division=0) for t in ths]
    best_th = ths[np.argmax(f1s)]
    y_pred = (y_prob >= best_th).astype(int)

    # 计算二分类指标
    metrics = {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "acc": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "th": best_th
    }
    return metrics

def train():
    graph_base_dir = "/home/doky/project/postgraduate/model/fine_tuning_llm_and_gnn/process_ast/output_graphs/four_model/blob"
    # 确保你的 jsonl 文件现在包含的是二分类标签
    train_json = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/blob_train_four_model.jsonl"
    eval_json = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/blob_eval_four_model.jsonl"
    vocab_path = "vocab.json"

    vocab_size = len(json.load(open(vocab_path)))

    # 核心参数
    num_classes = 1  # 二分类任务

    embed_dim = 512
    ffn_dim = 1024
    num_layers = 16
    num_heads = 16
    lr = 3e-5
    weight_decay = 1e-4
    batch_size = 4
    epochs = 50
    patience = 10
    max_nodes = 512
    device = torch.device("cuda")

    pretrained_embed_path = "/home/doky/project/postgraduate/model/multi_label_model/gnn/word2vec/word2vec_embeddings_512/pretrained_embeddings.pt"

    current_date = datetime.datetime.now().strftime("%Y%m%d")
    base_dir = os.path.join(os.getcwd(), current_date)
    counter = 1
    while True:
        save_dir = os.path.join(base_dir, f"run_binary_{counter:03d}")  # 修改了目录名
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            break
        counter += 1

    print(f"模型将保存在: {save_dir}")

    train_set = JavaGraphDataset(train_json, max_nodes=max_nodes, graph_base_dir=graph_base_dir,
                                 pretrained_embed_path=pretrained_embed_path)
    val_set = JavaGraphDataset(eval_json, max_nodes=max_nodes, graph_base_dir=graph_base_dir,
                               pretrained_embed_path=pretrained_embed_path)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = GraphormerClassifier(
        num_classes=num_classes,  # 传入 1
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        ffn_dim=ffn_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_pretrained=True
    ).to(device)

    # 更改损失函数
    criterion = nn.BCEWithLogitsLoss()

    nn.init.xavier_uniform_(model.classifier.weight)
    nn.init.zeros_(model.classifier.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode='max')

    # 调整训练和评估循环
    best_f1 = 0  # 监控 F1 分数
    patience_cnt = 0
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for data in pbar:
            data = {k: v.to(device) for k, v in data.items() if isinstance(v, torch.Tensor)}
            optimizer.zero_grad()
            model_output = model(data)

            # data["y"] 形状为 [B, 1], model_output 形状也为 [B, 1]
            labels = data["y"]
            loss = criterion(model_output, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 评估
        metrics = evaluate(val_loader, model, device)
        scheduler.step(metrics['f1'])  # 根据 F1 分数调整学习率

        print(
            f"Epoch {epoch:02d} | F1={metrics['f1']:.4f} Acc={metrics['acc']:.4f} Recall={metrics['recall']:.4f} Th={metrics['th']:.2f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            model_path = os.path.join(save_dir, "best_graphormer_binary.pt")
            torch.save(model.state_dict(), model_path)
            print(f"模型已保存到: {model_path}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("Early stop!")
                break

if __name__ == "__main__":
    train()