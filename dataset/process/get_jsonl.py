import json
import os
import pandas as pd

# # Java文件基础路径（假设names.txt中的name是相对此路径的）
# java_base_path = "../data/blob/smell/first_files/"
# # 输出JSONL文件路径
# output_path = "../data/blob/smell/has_smell.jsonl"
# # 读取Excel文件获取name和start_line、end_line
# df = pd.read_excel("../data/blob/smell/has_smell.xlsx")


output_paths = [
    "../data/blob/none/none_smell_v2.jsonl",
    "../data/blob/smell/has_smell_v2.jsonl",
    "../data/data_class/none/none_smell_v2.jsonl",
    "../data/data_class/smell/has_smell_v2.jsonl",
    "../data/feature_envy/none/none_smell_v2.jsonl",
    "../data/feature_envy/smell/has_smell_v2.jsonl",
    "../data/long_method/none/none_smell_v2.jsonl",
    "../data/long_method/smell/has_smell_v2.jsonl"
]

excel_paths = [
    "../data/blob/none/none_smell.xlsx",
    "../data/blob/smell/has_smell.xlsx",
    "../data/data_class/none/none_smell.xlsx",
    "../data/data_class/smell/has_smell.xlsx",
    "../data/feature_envy/none/none_smell.xlsx",
    "../data/feature_envy/smell/has_smell.xlsx",
    "../data/long_method/none/none_smell.xlsx",
    "../data/long_method/smell/has_smell.xlsx"
]

java_base_paths = [
    "../data/blob/none/first_files/",
    "../data/blob/smell/first_files/",
    "../data/data_class/none/first_files/",
    "../data/data_class/smell/first_files/",
    "../data/feature_envy/none/first_files/",
    "../data/feature_envy/smell/first_files/",
    "../data/long_method/none/first_files/",
    "../data/long_method/smell/first_files/"
]

labels = [
    0,
    1,
    0,
    2,
    0,
    3,
    0,
    4
]

i = 0
# 从lines和names中读取数据
def get_jsonl(start, end, name, out_file, label, idx):
    global i
    i+=1
    java_base_path = java_base_paths[idx]

    # 构建Java文件完整路径
    java_file_path = os.path.join(java_base_path, name)
        
        # 检查文件是否存在
    if not os.path.exists(java_file_path):
        print(f"文件不存在: {java_file_path}")
        return

        # 读取Java文件并提取指定行
    try:
        with open(java_file_path, "r", encoding="utf-8") as java_file:
            all_lines = java_file.readlines()
                
            # 确保行号在有效范围内
            start = max(1, min(start, len(all_lines)))
            end = max(start, min(end, len(all_lines)))
                
            # 提取指定行（注意行号从1开始，而列表索引从0开始）
            code_lines = [l.strip() for l in all_lines[start-1:end]]
            code = "\n".join(code_lines)
            if code == "404: Not Found":
                return
                
            # 构建JSON对象并写入JSONL文件
            record = {
                "code": code,
                "label": label,
                }
            out_file.write(json.dumps(record) + "\n")
                
    except Exception as e:
        print(f"处理文件 {java_file_path} 时出错: {str(e)}")


if "__main__" == __name__:
    idx = 0   

    # 读取Excel文件
    for excel_path, output_path, label in zip(excel_paths, output_paths, labels):
        df = pd.read_excel(excel_path)
        # 提取链接和文件名
        if 'start_line' in df.columns and 'end_line' in df.columns:
            start_lines = df['start_line'].tolist()
            end_lines = df['end_line'].tolist()
            link_data = df['link'].tolist()

            with open(output_path, 'w') as out_file:
                for line, start, end in zip(link_data, start_lines, end_lines):
                    # 移除末尾#L
                    parts = line.split("/#L")[0]
                    # 从删除后的link中获取下载的文件名
                    name = parts.split("/blob/", 1)[1].replace('/', '_')
                    name = name.split('_', 1)[1]
                    print(f"处理文件: {name}, 行范围: {start}-{end}")
                    get_jsonl(int(start), int(end), name, out_file, label, idx)
        idx += 1
    print(f"处理完成，共处理了 {i} 条数据")
