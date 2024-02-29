# 论文自动评阅项目

项目文件包括:

- README.md（描述项目文件）

- BERT.py (使用BERT模型根据论文评审意见对论文质量进行预测)
- Draw.py (根据聚类结果，绘制雷达图)

- GUI.py（图形化界面）

- MultipleLinearRegression.py(使用多元线性回归算法分析论文指标对论文总分影响权重的占比)

- NaiveBayes.py(使用朴素贝叶斯算法根据论文评审意见对论文质量进行预测)

- Predict.py（使用训练好的模型进行预测）

- Statistics.py（统计训练数据的分布情况：“优秀”、“良好”、“一般”、“较差”）

- WordFrequencyCount.py（使用Kmeans算法对训练数据进行聚类分析）

- 目录 "Data/" ：

  - cluster_keywords_scores.csv（聚类结果和关键词分数）

  - clustered_data.csv（带有聚类标签的原始数据）

  - paper_comments.csv（用于训练的数据集，包含“内容”、“评分”、“评价”、“不足之处“、”论文选题“、”文献综述“、”论文水平“、”论文写作“、”对论文熟悉程度“）

  - regression_coefficients_pie_chart.png（论文指标对论文总分影响权重的占比图）

  - stopwords.txt（停词表）