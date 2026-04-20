import json

def main():
    paths = [
        "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/feature_envy_train_four_model.jsonl",
        "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/feature_envy_eval_four_model.jsonl",
        "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/feature_envy_test_four_model.jsonl",
    ]
    output_paths = [
        "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/labels/feature_envy/train/labels.txt",
        "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/labels/feature_envy/eval/labels.txt",
        "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/dataset/gnn/labels/feature_envy/test/labels.txt",
    ]
    labels = [[],[],[]]
    i = 0
    for path in paths:
        with open(path) as f:
            data = f.readlines()
            for line in data:
                label = json.loads(line)["label"]
                labels[i].append(label)
        i = i + 1

    i = 0
    for output_path in output_paths:
        with open(output_path, "w") as f:
            for label in labels[i]:
                f.write(str(label) + "\n")
        i = i + 1

if __name__ == "__main__":
    main()
