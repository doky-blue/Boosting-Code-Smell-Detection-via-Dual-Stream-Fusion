import json
import random

# 使用示例
input_files = [
    '../data/blob/none/none_smell_v2.jsonl',
    '../data/blob/smell/has_smell_v2.jsonl',
    '../data/data_class/none/none_smell_v2.jsonl',
    '../data/data_class/smell/has_smell_v2.jsonl',
    '../data/feature_envy/none/none_smell_v2.jsonl',
    '../data/feature_envy/smell/has_smell_v2.jsonl',
    '../data/long_method/none/none_smell_v2.jsonl',
    '../data/long_method/smell/has_smell_v2.jsonl',
    ]

output_files = [
    '../data/blob/blob.jsonl',
    '../data/data_class/data_class.jsonl',
    '../data/feature_envy/feature_envy.jsonl',
    '../data/long_method/long_method.jsonl'
]

def merge_jsonl_files(input_files, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in input_files:
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line in infile:
                    # 验证是否为有效的JSON行（可选）
                    try:
                        json.loads(line)
                        outfile.write(line)
                    except json.JSONDecodeError:
                        print(f"跳过无效的JSON行在文件 {file_path}: {line.strip()}")

def merge_datas(none_smell_file, has_smell_file, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(none_smell_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    json.loads(line)
                    outfile.write(line)
                except json.JSONDecodeError:
                    print(f"跳过无效的JSON行在文件 {none_smell_file}: {line.strip()}")
        with open(has_smell_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    json.loads(line)
                    outfile.write(line)
                except json.JSONDecodeError:
                    print(f"跳过无效的JSON行在文件 {has_smell_file}: {line.strip()}")

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
    print(f"训练集: {train_end} 行 ({train_ratio*100:.0f}%)")
    print(f"验证集: {val_end - train_end} 行 ({val_ratio*100:.0f}%)")
    print(f"测试集: {total_lines - val_end} 行 ({(1 - train_ratio - val_ratio)*100:.0f}%)")

for i, output_file in enumerate(output_files):
    merge_datas(input_files[i*2], input_files[i*2+1], output_file)
    print(f"合并完成: {output_file}")

output_file = '../data/all_data.jsonl'
merge_jsonl_files(input_files, output_file)

# 使用示例
input_file = output_file
train_file = '../data/train_v2.jsonl'
val_file = '../data/eval_v2.jsonl'
test_file = '../data/test_v2.jsonl'

split_jsonl(input_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1)