import argparse
from datetime import datetime

import torch
import numpy as np
import json
import random
import os
import logging

from tqdm import tqdm
from transformers import RobertaTokenizer,  RobertaConfig, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from peft import LoraConfig, get_peft_model, TaskType

# CLASS_NAMES = ["none_smell", "blob", "data_class", "feature_envy", "long_method"]
# ID2LABEL = {i: label for i, label in enumerate(CLASS_NAMES)}
date = datetime.now().strftime("%Y-%m-%d")



class WeightedLossTrainer(Trainer):
    def __init__(self, weight, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits.float()  # 确保 logits 在同一设备上


        # 计算损失
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weight)
        loss = loss_fct(logits, labels)

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

    return dataset, label_encoder
# 定义计算指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
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
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
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
    
    #print arguments
    args = parser.parse_args()
    logging.basicConfig(
        filename=f"{date}_train.log",
        level=logging.INFO,  # 记录级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logging.info("device: %s, n_gpu: %s",device, args.n_gpu)

    #set seed
    set_seed(args.seed)

    # 加载模型
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    # config.id2label = ID2LABEL
    config.num_labels = 5

    model = RobertaForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config
    ).to(args.device)


    # 加载数据 - 确保返回测试集
    test_dataset, label_encoder = load_and_preprocess_data(args.test_data_file, tokenizer)  # 修改为直接加载测试集
    train_dataset, label_encoder = load_and_preprocess_data(args.train_data_file, tokenizer)  # 修改为直接加载测试集
    eval_dataset, label_encoder = load_and_preprocess_data(args.eval_data_file, tokenizer)  # 修改为直接加载测试集

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    output_dir = args.output_dir + f"/{date}"


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
    )

    # 初始化Trainer
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )

    # 13345:10403:860:974:410:698
    weight = torch.tensor(np.array(np.sqrt([
        (13345-10403)/10403,
        (13345-860)/860,
        (13345-974)/974,
        (13345-410)/410,
        (13345-698)/698
    ]), dtype=np.float32).tolist()).to(device)  # 权重

    # print("weight:", weight)

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    peft_trainer = WeightedLossTrainer(
        weight=weight,  # 权重
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,  # 计算指标
    )

    # 训练模型
    if args.do_train:
        logging.info("开始训练...")

        peft_trainer.train()
        peft_trainer.save_model()  # 保存模型
        tokenizer.save_pretrained(output_dir)  # 保存tokenizer
    
    if args.do_eval:
        logging.info("开始评估...")
        eval_results = peft_trainer.evaluate(eval_dataset=test_dataset)
        logging.info("评估结果: %s", eval_results)
        print(f"OPTUNA_METRIC:eval_f1:{eval_results['eval_f1']}")  # 关键行：输出需要优化的指标
        

if __name__ == "__main__":
    main()