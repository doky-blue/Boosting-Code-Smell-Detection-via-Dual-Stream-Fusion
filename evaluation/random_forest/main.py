import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def train_code_classifier(jsonl_file):
    # 1. 加载和准备数据
    data = load_jsonl(jsonl_file)
    df = pd.DataFrame(data)
    
    # 2. 特征提取
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        token_pattern=r'\b\w+\b',
        min_df=5,
        max_df=0.8
    )
    X = vectorizer.fit_transform(df['code'])
    y = df['label']
    
    # 3. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. 训练模型
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_classifier.fit(X_train, y_train)
    
    # 5. 评估模型
    y_pred = rf_classifier.predict(X_test)
    print(classification_report(y_test, y_pred))
    with open('classification_report.txt', 'w') as f:
        f.write(classification_report(y_test, y_pred))
    
    # 6. 保存模型
    joblib.dump(rf_classifier, 'random_forest_code_classifier.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    
    return rf_classifier, vectorizer

# 使用示例
if __name__ == "__main__":
    model, vectorizer = train_code_classifier('/home/doky/project/postgraduate/dataset/MLCQ/data/all_data_V2.jsonl')