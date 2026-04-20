import os
import glob

def main():
    smell_list = ["blob", "feature_envy", "long_method", "data_class"]
    types = ["train", "eval", "test"]

    for smell in smell_list:
        for type in types:
            SNIPPETS_DIR = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/snippets/{smell}/{type}"
            PROJECT_DIR = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/java_project/{smell}/{type}"

            if not os.path.exists(PROJECT_DIR):
                os.makedirs(PROJECT_DIR)

            file_paths = glob.glob(os.path.join(SNIPPETS_DIR, "*.java"))

            for i, file_path in enumerate(file_paths):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                # 简单的启发式判断，可以根据需要改进
                if content.startswith("class ") or content.startswith("public class "):
                    wrapper_class_name = f"MethodWrapper{i:04d}"
                    new_filename = f"{wrapper_class_name}.java"
                    new_path = os.path.join(PROJECT_DIR, new_filename)
                    with open(new_path, 'w', encoding='utf-8') as dest_f:
                        dest_f.write(content)
                else:
                    wrapper_class_name = f"MethodWrapper{i:04d}"
                    wrapped_content = f"class {wrapper_class_name} {{\n {content} \n}}"
                    new_filename = f"{wrapper_class_name}.java"
                    new_path = os.path.join(PROJECT_DIR, new_filename)
                    with open(new_path, 'w', encoding='utf-8') as dest_f:
                        dest_f.write(wrapped_content)

            print(f"Processing complete. Files are ready in '{PROJECT_DIR}'.")

if __name__ == "__main__":
    main()