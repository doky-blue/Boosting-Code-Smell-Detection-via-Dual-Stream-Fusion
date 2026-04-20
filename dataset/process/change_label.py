import json


'''
    smells = ["blob", "data_class", "feature_envy", "long_method"]
    paths = []
    output_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/eval.jsonl"

    for smell in smells:
        paths.append(f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/{smell}_eval_four_model.jsonl")

    for path in paths:
        with open(path) as f, \
            open(output_file, 'a') as w:
            for line in f:
                line = json.loads(line)
                w.write(json.dumps(line, ensure_ascii=False) + '\n')
'''


'''
    types = ["train", "eval", "test"]
    smells = ["blob", "data_class", "feature_envy", "long_method"]

    label = 1

    for smell in smells:
        for type in types:
            path = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/{smell}_{type}_four_model.jsonl"
            new_path = f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/new_{smell}_{type}_four_model.jsonl"
            with open(path, "r", encoding="utf-8") as f:
                data = f.readlines()

            with open(new_path, "w", encoding="utf-8") as f:
                for line in data:
                    item = json.loads(line)
                    if item["label"] != 0:
                        item['label'] = label
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
        label = label + 1
'''

if __name__ == '__main__':

    smells = ["blob", "data_class", "feature_envy", "long_method"]
    paths = []
    output_file = "/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/test.jsonl"

    for smell in smells:
        paths.append(f"/home/doky/project/postgraduate/dataset/MLCQ/data/four_model/{smell}_test_four_model.jsonl")

    for path in paths:
        with open(path) as f, \
            open(output_file, 'a') as w:
            for line in f:
                line = json.loads(line)
                w.write(json.dumps(line, ensure_ascii=False) + '\n')