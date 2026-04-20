import argparse
from datetime import datetime

import torch
import numpy as np
import json
import random
import os
import logging

from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.utils.class_weight import compute_class_weight

# CLASS_NAMES = ["none_smell", "blob", "data_class", "feature_envy", "long_method"]
# ID2LABEL = {i: label for i, label in enumerate(CLASS_NAMES)}
date = datetime.now().strftime("%Y-%m-%d")

import os

os.environ["WANDB_DISABLED"] = "true"

from torch import nn
from transformers import Trainer

import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (Tensor, optional): 类别权重，对应公式中的 alpha。
                                      如果提供了 class_weights，则直接传入即可。
            gamma (float): 聚焦参数，默认值为 2.0。值越大越关注难分样本。
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # 1. 计算 Cross Entropy Loss (不带 reduction，以便后续处理)
        # 如果有 alpha (class_weights)，在这里直接传入
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')

        # 2. 计算 pt (预测概率)
        pt = torch.exp(-ce_loss)
        # 注意：如果 ce_loss 已经被 alpha 加权，这里的 pt 计算在数学上并不严谨地等于原始概率，
        # 但在实践中，直接基于 ce_loss 计算 pt 并应用 focal term 是一种常见的简化实现。
        # 若要严谨实现：应先算无权重的 ce_loss 得到 pt，再手动乘 alpha。
        # 为了兼容您代码中已有的 class_weights，我们采用更严谨的写法：

        # --- 严谨写法开始 ---
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).view(-1)
        pt = log_pt.exp()

        # 计算 Focal Term: (1 - pt)^gamma
        focal_term = (1 - pt).pow(self.gamma)

        # 基础 Loss
        loss = -1 * focal_term * log_pt

        # 应用 Alpha (类别权重)
        if self.alpha is not None:
            # 获取对应 target 的权重
            weights = self.alpha.gather(0, targets.view(-1))
            loss = loss * weights
        # --- 严谨写法结束 ---

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        # 将权重移动到正确的设备
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        # 新增 gamma 参数
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # 前向传播
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # === 修改开始：使用 Focal Loss ===
        # 初始化 FocalLoss，传入现有的 class_weights 作为 alpha
        loss_fct = FocalLoss(alpha=self.class_weights, gamma=self.gamma)

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        # === 修改结束 ===

        return (loss, outputs) if return_outputs else loss

def load_and_preprocess_data(jsonl_path, tokenizer, max_length=512):
    """
    加载 jsonl 格式数据并预处理成 RoBERTa 可接受的格式

    Args:
        jsonl_path (str): jsonl 文件路径
        tokenizer (RobertaTokenizer): 用于文本编码
        max_length (int): 最大序列长度(默认512)

    Returns:
        dataset (Dataset): 包含 input_ids, attention_mask, labels 的数据集
        label_encoder (dict): 标签映射字典（如 {"label1": 0, "label2": 1}）
    """
    # ============ 1. 加载 jsonl 数据 ============
    codes = []
    raw_labels = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            codes.append(data["code"])  # 假设每行有 "text" 字段
            raw_labels.append(data["label"])  # 假设每行有 "label" 字段
    # print(raw_labels)

    # ============ 2. 标签编码 ============
    unique_labels = sorted(list(set(raw_labels)))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_labels = [label_encoder[label] for label in raw_labels]

    # ============ 新增：计算类别权重 ============
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(encoded_labels),
        y=encoded_labels
    )
    # 转换为 Tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    # =========================================

    # ============ 3. Tokenize 文本 ============
    tokenized_inputs = tokenizer(
        codes,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # ============ 4. 构建 Dataset ============
    dataset = Dataset.from_dict({
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": torch.tensor(encoded_labels),
    })

    return dataset, label_encoder, class_weights_tensor


# 定义计算指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro"),
    }


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # logging.set_verbosity_warning()

    parser = argparse.ArgumentParser()

    parser.add_argument("--type", default=None, type=str, required=True,
                        help="The code smell type that model detect.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a jsonl file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to evaluate the perplexity on (a jsonl file).")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.Preventing Exploding Gradients")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--focal_gamma", default=2.0, type=float,
                        help="Gamma parameter for Focal Loss.")

    # print arguments
    args = parser.parse_args()
    logging.basicConfig(
        filename=f"{date}_train.log",
        level=logging.INFO,  # 记录级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logging.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # set seed
    set_seed(args.seed)

    # 加载模型
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.num_labels = 2
    # config.id2label = ID2LABEL

    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config
    ).to(args.device)

    type = args.type
    # blob_train_four_model.jsonl
    test_data_file = args.test_data_file + f"{type}_test_four_model.jsonl"
    train_data_file = args.train_data_file + f"{type}_train_four_model.jsonl"
    eval_data_file = args.eval_data_file + f"{type}_eval_four_model.jsonl"

    # 加载数据 - 确保返回测试集
    test_dataset, label_encoder, _ = load_and_preprocess_data(test_data_file, tokenizer)
    eval_dataset, label_encoder, _ = load_and_preprocess_data(eval_data_file, tokenizer)

    train_dataset, label_encoder, train_weights = load_and_preprocess_data(train_data_file, tokenizer)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    output_dir = args.output_dir + f"/{type}" + f"/{date}"

    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,  # 输出目录
        num_train_epochs=args.num_train_epochs,  # 训练epoch数
        per_device_train_batch_size=args.train_batch_size,  # 每个设备的训练batch大小
        per_device_eval_batch_size=args.eval_batch_size,  # 每个设备的评估batch大小
        learning_rate=args.learning_rate,  # 学习率
        weight_decay=args.weight_decay,  # 权重衰减
        adam_epsilon=args.adam_epsilon,  # Adam优化器的epsilon值
        max_grad_norm=args.max_grad_norm,  # 最大梯度范数
        evaluation_strategy="epoch",  # 每个epoch评估一次
        save_strategy="epoch",  # 每个epoch保存一次模型
        logging_dir="./logs",  # 日志目录
        logging_steps=100,  # 日志记录步数
        warmup_ratio=0.1,  # 10% 的训练步数用于预热
        # ================== 修改此处 ==================
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="eval_f1",  # 使用宏平均F1作为判断标准
        greater_is_better=True,  # F1分数越高越好
        # ================================================================
    )

    # 初始化Trainer
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
    # print("weight:", weight)

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    # 早停
    from transformers import EarlyStoppingCallback
    from transformers import EarlyStoppingCallback
    peft_trainer = WeightedLossTrainer(  # 不再是 Trainer
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        class_weights=train_weights,  # 传入计算好的权重
        gamma=args.focal_gamma if hasattr(args, 'focal_gamma') else 2.0  # 传入 gamma
    )

    # 训练模型
    if args.do_train:
        logging.info("开始训练...")

        peft_trainer.train()
        peft_trainer.save_model()  # 保存模型
        tokenizer.save_pretrained(output_dir)  # 保存tokenizer

    if args.do_test:
        logging.info("开始评估...")
        eval_results = peft_trainer.evaluate(eval_dataset=test_dataset)
        logging.info("评估结果: %s", eval_results)
        print(f"OPTUNA_METRIC:eval_f1:{eval_results['eval_f1']}")  # 关键行：输出需要优化的指标


if __name__ == "__main__":
    main()