// export_single_cpg.sc

// 使用标准 main 方法接收参数数组
@main def exportCpg(cpgPath: String, graphmlPath: String) = {
    // 1. 检查文件是否存在
    if (!new java.io.File(cpgPath).exists()) {
        println(s"Error: CPG file not found at $cpgPath")
        sys.exit(1)
    }

    try {
        // 2. 尝试打开 CPG
        // Joern 4.x+ 推荐使用 openCpg()，它会返回一个 Option[Cpg]
        val project = openCpg(cpgPath)

        // 检查 CPG 是否成功打开
        if (project.isEmpty) {
            println(s"Error: Could not open CPG at $cpgPath")
            sys.exit(1)
        }

        val cpg = project.get.cpg

        // 3. 运行数据流分析（可选，但推荐）
        run.ossdataflow

        // 4. 修复导出方法：使用 newGraph + export。
        // cpg.all 是一个遍历器，遍历所有节点和边
        cpg.all.newGraph.export(graphmlPath)

        println(s"SUCCESS: Exported CPG from $cpgPath to $graphmlPath")

    } catch {
        case e: Exception =>
            // 捕获打开或导出过程中的任何异常
            println(s"ERROR: Failed to process $cpgPath. Details: ${e.getMessage}")
            // 确保发生错误时返回非零状态码
            sys.exit(1)
    }
    // 注意：已移除 closeCpg()，Joern 会在脚本结束时自动清理
}