import json
import random

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

def shuffle_jsonl(input_file, output_file):
    # 读取所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
     
    # 打乱行顺序
    random.shuffle(lines)
    
    # 写入新文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)

types = ['blob', 'data_class', 'feature_envy', 'long_method']

for type in types:
    input_file = f"/home/doky/project/postgraduate/dataset/MLCQ/data/{type}/{type}.jsonl"
    train_file = f"/home/doky/project/postgraduate/dataset/MLCQ/data/{type}/{type}_train.jsonl"
    eval_file = f"/home/doky/project/postgraduate/dataset/MLCQ/data/{type}/{type}_eval.jsonl"
    test = f"/home/doky/project/postgraduate/dataset/MLCQ/data/{type}/{type}_test.jsonl"
    split_jsonl(input_file, train_file, eval_file, test)

# 使用示例
# shuffle_jsonl('../data/all_data.jsonl', '../data/all_data_v2.jsonl')