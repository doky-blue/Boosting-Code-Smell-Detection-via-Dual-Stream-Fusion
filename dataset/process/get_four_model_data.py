import json

if __name__ == "__main__":
    smell_list = ["blob", "data_class", "feature_envy", "long_method"]
    files = [
        "/home/doky/project/postgraduate/dataset/MLCQ/data/train_v2.jsonl",
        "/home/doky/project/postgraduate/dataset/MLCQ/data/eval_v2.jsonl",
        "/home/doky/project/postgraduate/dataset/MLCQ/data/test_v2.jsonl"
    ]

    # 每个异味均生成一组数据
    for smell in smell_list:
        output_files = [
            f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/one_label/{smell}_train.jsonl",
            f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/one_label/{smell}_eval.jsonl",
            f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/one_label/{smell}_test.jsonl"
        ]

        # 每组数据集逐个处理
        for file, output_file in zip(files, output_files):
            with open(file, "r", encoding="utf-8") as f, \
                open(output_file, "w", encoding="utf-8") as output:
                data = f.readlines()
                for line in data:
                    item = json.loads(line)
                    if item["label"] != 0:
                        item["label"] = 1
                    output.write(json.dumps(item, ensure_ascii=False) + '\n')

