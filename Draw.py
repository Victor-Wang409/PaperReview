import matplotlib.pyplot as plt
import numpy as np
from math import pi

# 设置 Matplotlib 支持中文的字体
plt.rcParams['font.family'] = ['Arial Unicode MS']  # MacOS系统可用

# 假设我们已经有了每个聚类的关键词及其TF-IDF分数
cluster_keywords_scores = {
    0: {'论文结构合理': 12.6318, '写作规范': 4.6867, '结构合理': 6.1497, '逻辑清晰': 4.7532, '层次分明': 4.1879, '工作量饱满': 4.751, '图表规范': 3.5690, '同意答辩': 3.9035},
    1: {'论文结构合理': 25.2210, '写作规范': 22.3175, '结构合理': 15.7150, '逻辑清晰': 15.6814, '层次分明': 13.9406, '工作量饱满': 12.815, '图表规范': 11.3843,'同意答辩': 12.4978},
    2: {'论文结构合理': 23.6986, '写作规范': 7.5783, '结构合理': 11.5059, '逻辑清晰': 9.1830, '层次分明': 12.1330, '工作量饱满': 5.7251, '图表规范': 9.4168 ,'同意答辩': 14.8743},
    3: {'论文结构合理': 25.1942, '写作规范': 9.4747, '结构合理': 16.4827, '逻辑清晰': 14.4484, '层次分明': 19.7790, '工作量饱满': 13.7693, '图表规范': 11.4419 ,'同意答辩': 17.6020},
    4: {'论文结构合理': 5.8118, '写作规范': 5.0684, '结构合理': 7.1821, '逻辑清晰': 7.1821, '层次分明': 9.8395, '工作量饱满': 13.7651, '图表规范': 4.8979 ,'同意答辩': 17.5863},
}

# 提取所有关键词
keywords = list({key for cluster in cluster_keywords_scores.values() for key in cluster.keys()})

# 创建蜘蛛图数据
num_vars = len(keywords)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

for cluster_id, scores in cluster_keywords_scores.items():
    values = [scores.get(keyword, 0) for keyword in keywords]
    values += values[:1]  # 确保闭合
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'Cluster {cluster_id}')

# 关键词作为轴标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(keywords)

# 添加图例和标题
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
plt.title('Cluster Analysis with Spider Chart')

plt.show()
