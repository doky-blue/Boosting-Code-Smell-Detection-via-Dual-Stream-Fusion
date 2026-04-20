import pandas as pd

paths = [
    "../data/blob/",
    "../data/data_class/",
    "../data/feature_envy/",
    "../data/long_method/",
    ]

def get_link_and_name(path, links_file, names_file, lines_file):
    df = pd.read_excel(path)

    links = []
    link_data = []

    if 'link' in df.columns:
        link_data = df['link'].tolist()

    if 'start_line' in df.columns and 'end_line' in df.columns:
        start_lines = df['start_line'].tolist()
        end_lines = df['end_line'].tolist()

        with open(lines_file, 'w') as linesfile:
            for start, end in zip(start_lines, end_lines):
                linesfile.write(str(start) + '\t')
                linesfile.write(str(end) + '\n')

    with open(links_file, 'w') as linksfile, \
    open(names_file, 'w') as namesfile:

        for line in link_data:
            # 移除末尾#L
            parts = line.split("/#L")
            temp = parts[0]
            if temp not in links:
                links.append(temp)

        for line in links:
            # 从删除后的link中获取下载的文件名
            name = line.split("/blob/", 1)[1].replace('/', '_')
            name = name.split('_', 1)[1]
            namesfile.write(name + '\n')

            line = line.replace("github.com", "raw.githubusercontent.com")
            temp = line.split("/blob/")
            if len(temp) > 2:
                res = line.replace("/blob/", "/", 1)
            else:
                res = line.replace("/blob/", "/")
            linksfile.write(res + '\n')
        
    print(path + " done")


if "__main__" == __name__:
    for path in paths:
        none_smell_path = path + "none/none_smell.xlsx"
        none_smell_links_file = path + "none/links.txt"
        none_smell_names_file = path + "none/names.txt"
        none_smell_lines_file = path + "none/lines.txt"

        has_smell_path = path + "smell/has_smell.xlsx"
        has_smell_links_file = path + "smell/links.txt"
        has_smell_names_file = path + "smell/names.txt"
        has_smell_lines_file = path + "smell/lines.txt"

        get_link_and_name(none_smell_path, none_smell_links_file, none_smell_names_file, none_smell_lines_file)
        get_link_and_name(has_smell_path, has_smell_links_file, has_smell_names_file, has_smell_lines_file)
        