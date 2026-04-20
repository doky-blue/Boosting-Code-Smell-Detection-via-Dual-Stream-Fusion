import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# =========================================================================
# 步骤1: 模拟你的数据集和CodeBERT模型
# =========================================================================

# 创建一个模拟的多分类数据集（例如，4个类别）
# 假设你的二分类模型负责区分每个类别和其他类别
# 例如，模型1区分类别0 vs 其他, 模型2区分类别1 vs 其他, etc.
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=5,
                           n_redundant=5, n_clusters_per_class=1, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 这里，我们用简单的sklearn分类器来模拟你训练好的CodeBERT模型。
# 在实际项目中，你应该替换为你的CodeBERT模型实例。
# 假设每个模型都用作一个二分类器，用于区分一个特定类别 vs 其他所有类别。
# 注意：你的CodeBERT模型在训练时可能需要额外的包装来处理这种 One-vs-Rest 的情况。
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class DummyCodeBERTClassifier:
    """
    一个用于模拟你训练好的CodeBERT模型的类。
    你的实际模型可能有一个 predict_proba 方法。
    """

    def __init__(self, classifier_type):
        if classifier_type == 'svc':
            self.model = SVC(kernel='linear', probability=True, random_state=42)
        elif classifier_type == 'dt':
            self.model = DecisionTreeClassifier(random_state=42)
        elif classifier_type == 'nb':
            self.model = GaussianNB()
        elif classifier_type == 'mlp':
            self.model = MLPClassifier(random_state=42)

    def fit(self, X, y):
        # 你的实际模型可能不需要这一步，因为它们已经训练好了
        self.model.fit(X, y)

    def predict_proba(self, X):
        # 你的CodeBERT模型应该返回 (n_samples, n_classes) 的概率数组
        # 对于二分类，它可能返回 (n_samples, 2)
        # StackingClassifier 可以处理这个，它会取第2列的概率值
        return self.model.predict_proba(X)


# 为每个类别准备一个二分类模型
# 在真实场景中，这些模型是你用 (类别i vs 非类别i) 的数据训练出来的
# 这里为了演示方便，我们用简单方法模拟
class_0_y = (y_train == 0).astype(int)
class_1_y = (y_train == 1).astype(int)
class_2_y = (y_train == 2).astype(int)
class_3_y = (y_train == 3).astype(int)

# 你的四个 CodeBERT 二分类模型
model_0 = DummyCodeBERTClassifier('svc')
model_1 = DummyCodeBERTClassifier('dt')
model_2 = DummyCodeBERTClassifier('nb')
model_3 = DummyCodeBERTClassifier('mlp')

# 训练你的基础模型（在真实项目中，它们已经训练好了）
model_0.fit(X_train, class_0_y)
model_1.fit(X_train, class_1_y)
model_2.fit(X_train, class_2_y)
model_3.fit(X_train, class_3_y)

# =========================================================================
# 步骤2: 实现 Stacking 分类器
# =========================================================================

# 将你的模型封装成 estimators 列表
# StackingClassifier 需要一个 (name, estimator) 的元组列表
# 注意：这里我们直接使用已经训练好的模型实例，而不是类
estimators = [
    ('codebert_model_0', model_0.model),
    ('codebert_model_1', model_1.model),
    ('codebert_model_2', model_2.model),
    ('codebert_model_3', model_3.model),
]

# 定义元模型（最终的分类器）
# 推荐使用逻辑回归作为起点
final_estimator = LogisticRegression(solver='liblinear')

# 定义 Stacking 分类器
# `cv=5` 表示使用 5 折交叉验证来生成元模型的训练数据，防止数据泄露
# `stack_method='predict_proba'` 表示使用基础模型的概率输出作为元模型的特征
# 这比直接使用类别标签 `predict` 效果更好
stacked_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    stack_method='predict_proba',
    cv=5
)

# =========================================================================
# 步骤3: 训练和预测
# =========================================================================

# 训练 stacking 模型
# StackingClassifier 会自动处理基础模型的训练和预测，然后训练元模型
print("开始训练 Stacking 模型...")
stacked_clf.fit(X_train, y_train)
print("训练完成。")

# 使用 Stacking 模型进行预测
stacked_predictions = stacked_clf.predict(X_test)

# 评估 Stacking 模型的性能
stacked_accuracy = accuracy_score(y_test, stacked_predictions)
print(f"\nStacking 模型的最终准确率: {stacked_accuracy:.4f}")

# 作为对比，计算其中一个基础模型的性能（例如，SVC）
# 注意，基础模型本身是二分类的，直接用于多分类预测会很差
# 这里只作为参考
svc_predictions = stacked_clf.estimators_[0].predict(X_test)
# svc_accuracy = accuracy_score(y_test, svc_predictions)
# print(f"其中一个基础模型（SVC）在多分类任务上的准确率: {svc_accuracy:.4f}")