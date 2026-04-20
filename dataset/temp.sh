#! /bin/bash

# 指定文本文件路径
link_path='links.txt'
filename_path='names.txt'

exec 3< "$link_path"
exec 4< "$filename_path"

# 逐行读取文件并保存每行内容到变量
while read -r line1 <&3 && read -r line2 <&4; do
    # 将每行内容保存到变量
    link="$line1"
    name="$line2"

    # echo "curl --max-time 300 -o $name -L $link"

    if curl --max-time 300 -o ./first_files/$name -L $link; then
        echo "Download successful for $name"
    else
        echo "$link" >> error_link.txt
        echo "$name" >> error_filename.txt
    fi

done

exec 3<&-
exec 4<&-