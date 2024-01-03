import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import numpy as np

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')
model.eval()

# 加载数据
data_path = './Data/data.csv'  # 替换为您的文件路径
data = pd.read_csv(data_path)
texts = data['内容'].to_list()

# 建立停词表
stopwords = []
with open("./Data/stopwords.txt", encoding="utf-8") as f:
    lines = f.readlines()
    for i in lines:
        line = i.strip()
        stopwords.append(line)
# 对停用词列表去重
stopwords = list(set(stopwords))

# 获取每个文本的嵌入向量
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length", stopwords=stopwords)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

embeddings = np.vstack([get_embedding(text) for text in texts])

# 应用 K-means 聚类
num_clusters = 5  # 可以根据需要调整聚类的数量
kmeans = KMeans(n_clusters=num_clusters,  verbose=1)
kmeans.fit(embeddings)
clusters = kmeans.labels_

# 提取每个聚类的关键词及其 TF-IDF 分数
cluster_keywords_scores = {}
for cluster_id in range(num_clusters):
    cluster_texts = [texts[i] for i in range(len(texts)) if clusters[i] == cluster_id]

    # 应用 TF-IDF 向量化器
    vectorizer = TfidfVectorizer(max_features=100, stop_words=stopwords)  # 调整 max_features 以控制关键词数量
    X = vectorizer.fit_transform(cluster_texts)

    # 提取特征名称（关键词）及其 TF-IDF 分数
    feature_names = vectorizer.get_feature_names_out()
    sorted_items = sorted(zip(feature_names, X.sum(axis=0).tolist()[0]), key=lambda x: -x[1])

    # 获取前 N 个关键词及其分数
    top_keywords_scores = [(item[0], item[1]) for item in sorted_items[:20]]  # 调整以控制提取的关键词数量
    cluster_keywords_scores[cluster_id] = top_keywords_scores

# 输出聚类及其关键词和分数
for cluster_id, keywords_scores in cluster_keywords_scores.items():
    print(f"Cluster {cluster_id}:")
    for keyword, score in keywords_scores:
        print(f"    {keyword}: {score:.4f}")


# 创建保存结果的 DataFrame
cluster_results = pd.DataFrame(columns=['Cluster', 'Keyword', 'Score'])

for cluster_id, keywords_scores in cluster_keywords_scores.items():
    for keyword, score in keywords_scores:
        cluster_results = cluster_results._append({'Cluster': cluster_id, 'Keyword': keyword, 'Score': score}, ignore_index=True)

# 将聚类标签添加到原始数据中
data['Cluster'] = clusters

# 保存聚类结果和原始数据
cluster_results.to_csv('cluster_keywords_scores.csv', index=False)
data.to_csv('clustered_data.csv', index=False)

print("聚类结果和关键词分数已保存至 'cluster_keywords_scores.csv'")
print("带有聚类标签的原始数据已保存至 'clustered_data.csv'")