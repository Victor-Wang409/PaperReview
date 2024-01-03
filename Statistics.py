import pandas as pd
import matplotlib.pyplot as plt


# 设置 Matplotlib 支持中文的字体
plt.rcParams['font.family'] = ['Arial Unicode MS']  # MacOS系统可用

# 文件路径
file_path = './Data/data.csv'

# 读取数据
data = pd.read_csv(file_path)

# 统计评价
evaluation_counts = data['评价'].value_counts()

# 绘制柱状图
plt.figure(figsize=(10, 6))
bars = plt.bar(evaluation_counts.index, evaluation_counts.values)
evaluation_counts.plot(kind='bar')

# 在每一条柱状图上添加文本标签
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.title('评价统计')
plt.xlabel('评价类型')
plt.ylabel('数量')
plt.xticks(rotation=45)
plt.show()

# 自定义颜色
colors = ['#66b3ff','#99ff99','#ff9999','#ffcc99']

# 使用前面的代码绘制饼图，并应用新的颜色列表
plt.figure(figsize=(8, 8))
plt.pie(evaluation_counts, labels=evaluation_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('评价分布')
plt.show()