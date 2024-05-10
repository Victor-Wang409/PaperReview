import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC  # 导入支持向量机分类器
from sklearn.metrics import accuracy_score
from joblib import dump

def load_data(file_path):
    # 从CSV文件加载数据
    comments_data = pd.read_csv(file_path, encoding="gbk")
    data = comments_data["内容"].tolist()
    labels = comments_data["评价"].tolist()
    return data, labels

def load_stop_words(file_path):
    # 从文件加载停用词，并去除重复项
    with open(file_path, encoding="utf-8") as f:
        stopwords = [line.strip() for line in f]
    return list(set(stopwords))

def preprocess_data(data):
    # 预处理数据，这里可以加入更多的处理逻辑
    return data

def train_model(data, labels, stopwords):
    # 使用停用词训练SVM模型
    vectorizer = CountVectorizer(stop_words=stopwords)
    data_vec = vectorizer.fit_transform(data)
    model = SVC(kernel='linear')  # 使用线性核
    model.fit(data_vec, labels)
    return model, vectorizer

def evaluate_model(model, vectorizer, test_data, test_labels):
    # 评估模型准确度
    test_data_vec = vectorizer.transform(test_data)
    predictions = model.predict(test_data_vec)
    print("准确度:", accuracy_score(test_labels, predictions))

def predict(model, vectorizer, new_data):
    # 使用模型进行预测
    new_data_vec = vectorizer.transform(new_data)
    return model.predict(new_data_vec)

def save_model(model, vectorizer, model_path):
    # 保存模型和向量器到指定路径
    os.makedirs(model_path, exist_ok=True)
    dump(model, os.path.join(model_path, 'svm_model.joblib'))
    dump(vectorizer, os.path.join(model_path, 'svm_vectorizer.joblib'))

def main():
    # 主函数，执行模型训练和评估流程
    data, labels = load_data("./Data/paper_comments.csv")
    stopwords = load_stop_words("./Data/stopwords.txt")
    processed_data = preprocess_data(data)
    train_data, test_data, train_labels, test_labels = train_test_split(processed_data, labels, test_size=0.2)
    model, vectorizer = train_model(train_data, train_labels, stopwords)
    evaluate_model(model, vectorizer, test_data, test_labels)
    save_model(model, vectorizer, './model')
    new_data = ["这篇论文非常有创新性"]
    print("预测:", predict(model, vectorizer, new_data))

main()
