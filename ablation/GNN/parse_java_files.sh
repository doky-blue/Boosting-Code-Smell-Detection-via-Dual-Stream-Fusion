#!/bin/bash

# --- 配置 ---
# 存放你的 Java 文件的输入文件夹
INPUT_DIR="/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/java_project/long_method/train/"
# 存放生成的 CPG 文件的输出文件夹
OUTPUT_DIR="/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/cpg/long_method/train/"
# --- 结束配置 ---

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录 '$INPUT_DIR' 不存在。"
    exit 1
fi

# 创建输出目录，如果它不存在的话
mkdir -p "$OUTPUT_DIR"

echo "开始处理 Java 文件..."

# 查找所有 .java 文件并逐个处理
# 使用 find 命令可以递归查找所有子目录中的 java 文件
find "$INPUT_DIR" -type f -name "*.java" | while read java_file; do
    # 从完整路径中获取不带扩展名的文件名
    # 例如: /path/to/MyClass.java -> MyClass
    base_name=$(basename "$java_file" .java)

    # 构造输出 CPG 文件的完整路径
    # 例如: /path/to/cpg_files/MyClass.cpg.bin
    output_cpg_file="$OUTPUT_DIR/${base_name}.cpg.bin"

    echo "-----------------------------------------------------"
    echo "正在处理: $java_file"
    echo "输出到:   $output_cpg_file"

    # 运行 joern-parse 命令
    joern-parse "$java_file" --output "$output_cpg_file"
done

echo "-----------------------------------------------------"
echo "所有文件处理完毕！"