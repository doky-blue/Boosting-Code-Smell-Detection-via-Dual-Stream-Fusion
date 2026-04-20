#!/bin/bash
# --- 配置 ---
# 设置存放.cpg 或.bin 文件的目录
CPG_DIRECTORY="/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/cpg/long_method/train/"

# 设置用于存放导出的 GraphML 文件的根目录
OUTPUT_DIRECTORY="/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/graphML/long_method/train/"
# --- 结束配置 ---

# 检查输入目录是否存在
if [ ! -d "$CPG_DIRECTORY" ]; then
  echo "错误: CPG 目录 '$CPG_DIRECTORY' 不存在。"
  exit 1
fi

# 创建输出目录（如果它不存在）
mkdir -p "$OUTPUT_DIRECTORY"

echo "开始批量转换..."

# 循环遍历指定目录中的每个.cpg 和.bin 文件
for cpg_file in "$CPG_DIRECTORY"/*.cpg "$CPG_DIRECTORY"/*.bin; do
  # 检查是否找到匹配的文件，以防目录为空
  [ -f "$cpg_file" ] || continue

  echo "正在处理文件: $cpg_file"

  # 提取不带扩展名的文件名
  base_name=$(basename "$cpg_file")
  filename_no_ext="${base_name%.*}"

  # 为每个文件创建一个唯一的输出子目录
  specific_output_dir="$OUTPUT_DIRECTORY/$filename_no_ext"

  # 运行 joern-export 命令
  # --repr=all 确保导出完整的图
  # --format=graphml 指定输出格式
  /home/doky/project/postgraduate/joern/joern-cli/joern-export "$cpg_file" --out "$specific_output_dir" --repr=all --format=graphml

  echo "已将 GraphML 导出到: $specific_output_dir"
  echo "---------------------------------"
done

echo "所有文件处理完毕。"