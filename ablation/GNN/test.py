import networkx as nx

# 假设导出的文件位于 'out/cpg.bin/graph.graphml'
graphml_file_path = '/home/doky/workspace/MethodWrapper0000.cpg.bin/out/export.xml'

try:
    # 从 GraphML 文件中读取图
    G = nx.read_graphml(graphml_file_path)

    # 现在您可以使用 NetworkX 对图 G 进行操作
    print(f"图已成功加载。")
    print(f"节点数量: {G.number_of_nodes()}")
    print(f"边的数量: {G.number_of_edges()}")

    # 示例：遍历前5个节点并打印它们的属性
    for node_id, node_data in list(G.nodes(data=True))[:5]:
        print(f"\n节点 ID: {node_id}")
        print("属性:")
        for key, value in node_data.items():
            print(f"  {key}: {value}")

except FileNotFoundError:
    print(f"错误：在路径 '{graphml_file_path}' 未找到文件。请确保路径正确。")
except Exception as e:
    print(f"加载图时发生错误: {e}")