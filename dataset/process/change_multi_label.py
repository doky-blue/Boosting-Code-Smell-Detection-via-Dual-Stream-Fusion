import json

def change_label(input_file, output_file):
    """
    读取JSONL文件，修改label值后保存为新文件

    :param input_file: 输入JSONL文件路径
    :param output_file: 输出JSONL文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                # 解析JSON对象
                obj = json.loads(line)
                # 修改label值
                if 'label' in obj:
                    label = [0, 0, 0, 0, 0]
                    label[obj['label']] = 1
                    obj['label'] = label
                    # obj['label'] = modification_func(obj['label'])
                    pass

                # 写入新文件
                outfile.write(json.dumps(obj, ensure_ascii=False) + '\n')
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {line}\nError: {e}")

if __name__ == '__main__':
    input_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/test_v2.jsonl"
    output_file = "/home/doky/project/postgraduate/dataset/MLCQ/multi_label/test.jsonl"

    change_label(input_file, output_file)