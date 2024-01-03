import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 数据和标签
data = []
# data = ""
labels = []

def data_loader():
    paper_comments = pd.read_csv("./Data/paper_comments.csv", encoding="gbk")
    paper_content = paper_comments["内容"]
    paper_labels = paper_comments["评价"]
    # print(paper_content.head())
    # print(paper_labels.head())
    for c in paper_content:
        data.append(c)
    for l in paper_labels:
        labels.append(l)
    print(data)
    print(labels)

# 建立停词表
def stop_words():
    stopwords = []
    with open("./Data/stopwords.txt", encoding="utf-8") as f:
        lines = f.readlines()
        for i in lines:
            line = i.strip()
            stopwords.append(line)

    # 对停用词列表去重
    stopwords = list(set(stopwords))
    return stopwords

def preprocess_data(data):
    # 这里可以添加更多的预处理步骤
    return data

def vectorize_data(data):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(data), vectorizer

def train_model(data_train, labels_train):
    # 向量化文本数据
    stopwords = stop_words()
    vectorizer = CountVectorizer(stop_words=stopwords)
    data_train_vec = vectorizer.fit_transform(data_train)

    # 训练朴素贝叶斯模型
    model = MultinomialNB()
    model.fit(data_train_vec, labels_train)

    return model, vectorizer

def evaluate_model(model, vectorizer, data_test, labels_test):
    # 向量化文本数据
    data_test_vec = vectorizer.transform(data_test)

    predictions = model.predict(data_test_vec)
    print("Accuracy:", accuracy_score(labels_test, predictions))

def predict(model, vectorizer, new_data):
    new_data_vec = vectorizer.transform(new_data)
    return model.predict(new_data_vec)


def nbm(data, labels):
    # 数据预处理
    processed_data = preprocess_data(data)

    data_loader()

    # 将数据划分为训练集和测试集
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)

    # 训练模型
    model, vectorizer = train_model(data_train, labels_train)

    # 评估模型
    evaluate_model(model,vectorizer, data_test, labels_test)

    # 使用模型进行预测（示例）
    new_data = ["这篇论文非常有创新性"]
    print("Prediction:", predict(model, vectorizer, new_data))

nbm(data, labels)
