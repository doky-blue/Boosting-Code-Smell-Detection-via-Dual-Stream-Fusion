# tune_hyperparameters.py
import optuna
import subprocess
import sys
from tqdm import tqdm

class TqdmCallback:
    def __init__(self, total_trials):
        self.pbar = tqdm(total=total_trials, desc="Optimizing", unit="trial")
    
    def __call__(self, study, trial):
        self.pbar.update(1)
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.pbar.set_postfix({"Best F1": study.best_value})

def objective(trial):
    # 超参数搜索空间
    params = {
        "learning_rate": trial.suggest_float("lr", 1e-6, 1e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0, 0.1),
        "epochs": trial.suggest_int("epochs", 50, 150, step=50),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-8, 1e-6),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0, 1.0)
    }

    # 构建命令（全部转为字符串）
    cmd = [
        sys.executable, "main.py",  # 使用当前Python解释器
        "--learning_rate", f"{params['learning_rate']:.6f}",
        "--train_batch_size", str(params["batch_size"]),
        "--weight_decay", str(params["weight_decay"]),
        "--num_train_epochs", str(params["epochs"]),
        "--adam_epsilon", f"{params['adam_epsilon']:.6f}",
        "--max_grad_norm", f"{params['max_grad_norm']:.6f}",
        # 其他固定参数...
        "--do_train",
        "--do_eval",
        "--do_test",
        "--output_dir", f"./output/output_trail_{trial.number}",
        "--model_name_or_path", "/home/doky/Llama/codebert-base",
        "--train_data_file", "/home/doky/project/postgraduate/dataset/MLCQ/data/train_v2.jsonl",
        "--eval_data_file", "/home/doky/project/postgraduate/dataset/MLCQ/data/eval_v2.jsonl",
        "--test_data_file", "/home/doky/project/postgraduate/dataset/MLCQ/data/test_v2.jsonl",
        "--eval_batch_size", "16",
    ]

    # 运行训练
    try:
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8"
        )
        
        # 检查是否成功
        if process.returncode != 0:
            raise RuntimeError(f"训练失败:\n{process.stderr}")

        # 解析评估结果（适配您的日志格式）
        for line in process.stdout.split("\n"):
            if "OPTUNA_METRIC:eval_f1:" in line:  # 完全匹配标记
                return float(line.split(":")[-1])  # 直接取最后一个冒号后的值
        raise ValueError("未找到评估结果")
    except Exception as e:
        print(f"试验 {trial.number} 失败: {str(e)}")
        raise optuna.TrialPruned()

if __name__ == "__main__":

    n_trials = 10  # 总试验次数

    # 创建study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # 创建Study并添加tqdm回调
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[TqdmCallback(n_trials)]  # 关键：添加进度条回调
    )
    
    # 输出结果
    print("最佳参数:", study.best_params)
    print("最佳F1分数:", study.best_value)
