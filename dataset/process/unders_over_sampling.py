import json
import random

import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 加载JSONL数据集
def load_data(filepath):
    """加载JSONL格式的代码异味数据集"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 2. 数据预处理
def preprocess_data(df):
    """提取特征和标签，保留原始代码"""
    print("原始数据分布:", Counter(df['label']))
    
    # TF-IDF特征提取
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words=None  # 对于代码数据通常不需要停用词
    )
    X = vectorizer.fit_transform(df['code'])
    y = df['label'].values
    codes = df['code'].values
    
    return X, y, codes, vectorizer

# 3. 混合采样（保留原始代码）
def hybrid_sampling_with_code(X, y, codes):
    """执行过采样+欠采样，并保留对应的原始代码"""
    
    # ---- 第一步：过采样少数类 ----
    over = RandomOverSampler(
        sampling_strategy={
            1: 1760,         # 目标样本数
            2: 1948,
            3: 820,
            4: 1396    # 根据您的实际标签调整
        },
        random_state=42
    )
    X_over, y_over = over.fit_resample(X, y)
    
    # 获取过采样后的代码（通过采样器返回的索引）
    over_indices = over.sample_indices_
    codes_over = [codes[i] for i in over_indices]
    
    print("过采样后分布:", Counter(y_over))
    
    # ---- 第二步：欠采样多数类 ----
    under = RandomUnderSampler(
        sampling_strategy={0: 5200},  # 保留4000个无气味样本
        random_state=42
    )
    X_res, y_res = under.fit_resample(X_over, y_over)
    
    # 获取最终代码（通过欠采样的索引）
    under_indices = under.sample_indices_
    resampled_codes = [codes_over[i] for i in under_indices]
    
    print("最终分布:", Counter(y_res))
    
    return X_res, y_res, resampled_codes

# 4. 保存处理后的数据集
def save_resampled_data(output_path, y_resampled, resampled_codes):
    """保存为JSONL格式"""
    with open(output_path, 'w') as f:
        for code, label in zip(resampled_codes, y_resampled):
            f.write(json.dumps({
                'code': code,
                'label': int(label)
            }) + '\n')


def split_jsonl(input_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1):
    """
    将JSONL文件按比例分割为三个文件

    参数:
        input_file: 输入JSONL文件路径
        train_file: 训练集输出文件路径
        val_file: 验证集输出文件路径
        test_file: 测试集输出文件路径
        train_ratio: 训练集比例 (默认0.8)
        val_ratio: 验证集比例 (默认0.1)
        # 测试集比例自动计算为 1 - train_ratio - val_ratio
    """
    # 读取所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 打乱顺序以确保随机分布
    random.shuffle(lines)

    total_lines = len(lines)
    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)

    # 写入训练集
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:train_end])

    # 写入验证集
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[train_end:val_end])

    # 写入测试集
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[val_end:])

    print(f"分割完成: 总行数 {total_lines}")
    print(f"训练集: {train_end} 行 ({train_ratio * 100:.0f}%)")
    print(f"验证集: {val_end - train_end} 行 ({val_ratio * 100:.0f}%)")
    print(f"测试集: {total_lines - val_end} 行 ({(1 - train_ratio - val_ratio) * 100:.0f}%)")

# 主流程
def main():
    # 文件路径配置
    input_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/all_data.jsonl"
    output_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/all_data_V3.jsonl"
    train_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/train_V3.jsonl"
    eval_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/eval_V3.jsonl"
    test_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/test_V3.jsonl"

    # 1. 加载数据
    df = load_data(input_file)
    
    # 2. 预处理
    X, y, codes, vectorizer = preprocess_data(df)
    
    # 3. 执行混合采样
    X_res, y_res, resampled_codes = hybrid_sampling_with_code(X, y, codes)
    
    # 4. 保存结果
    save_resampled_data(output_file, y_res, resampled_codes)
    print(f"处理后的数据已保存至: {output_file}")

    split_jsonl(output_file, train_file, eval_file, test_file ,train_ratio=0.8, val_ratio=0.1)

if __name__ == "__main__":
    main()
