import os
import xml.etree.ElementTree as ET

smells = ["blob", "data_class", "feature_envy", "long_method"]
types = ["eval", "test", "train"]

def extract_labels_from_file(file_path, namespaces):
    """
    从单个 GraphML 文件中提取所有唯一的 "labelV" 标签。

    Args:
        file_path (str): .xml 文件的路径。
        namespaces (dict): 用于 XML 解析的命名空间。

    Returns:
        set: 在此文件中找到的唯一标签的集合。
    """
    local_labels = set()
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 查找所有 <node> 元素
        nodes = root.findall('.//g:node', namespaces)

        for node in nodes:
            # 在每个 <node> 中查找 <data key="labelV">
            label_data = node.find("g:data[@key='labelV']", namespaces)

            if label_data is not None and label_data.text:
                local_labels.add(label_data.text.strip())

    except ET.ParseError as e:
        print(f"警告：跳过文件 {file_path} (XML 解析错误): {e}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")

    return local_labels


def find_all_unique_labels(directory_path, output_path):
    """
    遍历指定目录中的所有 .xml 文件，并收集所有唯一的 "labelV" 标签。

    Args:
        directory_path (str): 包含 .xml 文件的目录路径。
    """
    all_unique_labels = set()

    # Joern 的 GraphML 文件使用命名空间
    namespaces = {'g': 'http://graphml.graphdrawing.org/xmlns'}

    if not os.path.isdir(directory_path):
        print(f"错误：'{directory_path}' 不是一个有效的目录。")
        return

    print(f"开始处理目录: {directory_path}\n")

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        if filename.endswith(".xml"):
            file_path = os.path.join(directory_path, filename)
            print(f"正在解析: {filename}...")
            labels_from_file = extract_labels_from_file(file_path, namespaces)
            all_unique_labels.update(labels_from_file)

    print("\n--- 处理完成 ---")

    if not all_unique_labels:
        print("未找到任何标签。请检查文件是否正确，以及 'labelV' 是否存在。")
        return

    print("在所有文件中找到的所有唯一 CPG 节点标签:")
    # 排序后输出
    with open(output_path, "a+", encoding="utf-8") as f:
        for label in sorted(list(all_unique_labels)):
            f.writelines(label + "\n")

def merge_label(label_type, output_path):
    unique_labels = set()
    output_path = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/node_labels.txt"
    for smell in smells:
        for type in types:
            labels_path = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/graphML/{smell}/{type}/node_labels.txt"
            with open(labels_path, 'r') as f:
                labels = f.readlines()
                for label in labels:
                    unique_labels.add(label.strip())

    with open(output_path, "w", encoding="utf-8") as f:
        for label in sorted(list(unique_labels)):
            f.writelines(label + "\n")

if __name__ == "__main__":
    # for smell in smells:
    #     for type in types:
    #         DIRECTORY_PATH = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/graphML/{smell}/{type}"  # <-- 修改这里
    #         OUTPUT_PATH = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/graphML/{smell}/{type}/node_labels.txt"
    #         find_all_unique_labels(DIRECTORY_PATH, OUTPUT_PATH)

    merge_label()