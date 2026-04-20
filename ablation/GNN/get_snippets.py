import json

def main():

    train_java_path = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/long_method_train_four_model.jsonl"
    eval_java_path = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/long_method_eval_four_model.jsonl"
    test_java_path = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/long_method_test_four_model.jsonl"
    # eval_java_path = "/home/doky/project/postgraduate/dataset/MLCQ/data/eval_v2.jsonl"
    # test_java_path = "/home/doky/project/postgraduate/dataset/MLCQ/data/test_v2.jsonl"

    num = 0
    with open(train_java_path, "r") as f:
        all_data = f.readlines()
        for line in all_data:
            output_path = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/snippets/long_method/train/snippets_{num}.java"
            data = json.loads(line)
            code = data["code"]
            with open(output_path, "w") as output_file:
                output_file.write(code)
            num = num + 1

    num = 0
    with open(eval_java_path, "r") as f:

        all_data = f.readlines()
        for line in all_data:
            output_path = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/snippets/long_method/eval/snippets_{num}.java"
            data = json.loads(line)
            code = data["code"]
            with open(output_path, "w") as output_file:
                output_file.write(code)
            num = num + 1

    num = 0
    with open(test_java_path, "r") as f:
        all_data = f.readlines()
        for line in all_data:
            output_path = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/snippets/long_method/test/snippets_{num}.java"
            data = json.loads(line)
            code = data["code"]
            with open(output_path, "w") as output_file:
                output_file.write(code)
            num = num + 1

if __name__ == "__main__":
    main()