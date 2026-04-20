import os
import glob

def main():

    # --- 配置 ---
    SNIPPETS_DIR = "snippets/train"  # 存放不完整代码片段的目录
    PROJECT_DIR = "java_project/train"  # 处理后用于 Joern 分析的目录
    FILE_PATTERN = "*.java"  # 要处理的文件模式

    # --- 脚本 ---
    if not os.path.exists(PROJECT_DIR):
        os.makedirs(PROJECT_DIR)

    # 遍历所有 Java 文件片段
    file_paths = glob.glob(os.path.join(SNIPPETS_DIR, FILE_PATTERN))

    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # 简单判断内容是否已经是完整的类
        # 注意：这个判断非常初级，可能需要根据你的实际情况调整
        if content.startswith("class ") or content.startswith("public class ") \
                or content.startswith("interface ") or content.startswith("public interface "):

            # 认为是类文件，直接复制
            wrapper_class_name = f"MethodWrapper{i:04d}"
            new_filename = f"{wrapper_class_name}.java"
            new_path = os.path.join(PROJECT_DIR, new_filename)

            with open(new_path, 'w', encoding='utf-8') as dest_f:
                dest_f.write(content)
            print(f"Copied class file: {file_path} -> {new_path}")

        else:
            # 认为是方法片段，进行包装
            wrapper_class_name = f"MethodWrapper{i:04d}"
            wrapped_content = f"""
    class {wrapper_class_name} {{
        {content}
    }}
    """
            new_filename = f"{wrapper_class_name}.java"
            new_path = os.path.join(PROJECT_DIR, new_filename)
            with open(new_path, 'w', encoding='utf-8') as dest_f:
                dest_f.write(wrapped_content)
            print(f"Wrapped method snippet: {file_path} -> {new_path}")

    print(f"\nProcessing complete. All files are ready in '{PROJECT_DIR}' directory.")

if __name__ == "__main__":
    main()