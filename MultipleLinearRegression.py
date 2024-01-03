import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


data_labels = []
data_tst = []
data_lr = []
data_pl = []
data_tw = []

paper_comments = pd.read_csv("./Data/paper_comments_1.csv", encoding="gb2312")
paper_labels = paper_comments["评分"]
paper_tst = paper_comments["论文选题"]
paper_lr = paper_comments["文献综述"]
paper_pl = paper_comments["论文水平"]
paper_tw = paper_comments["论文写作"]

for l in paper_labels:
    data_labels.append(l)

for t in paper_tst:
    data_tst.append(t)

for l in paper_lr:
    data_lr.append(l)

for p in paper_pl:
    data_pl.append(p)

for t in paper_tw:
    data_tw.append(t)

data = pd.DataFrame({
    '论文选题': data_tst,
    '文献综述': data_lr,
    '论文水平': data_pl,
    '论文写作': data_tw,
    '评分': data_labels
})
print(data)

# 将分类数据转换为数值
def convert_grade_to_score(grade):
    if grade == '优秀':
        return 95
    elif grade == '良好':
        return 82
    elif grade == '一般':
        return 67
    elif grade == '不合格':
        return 30
    else:
        return 0

data['论文选题'] = data['论文选题'].apply(convert_grade_to_score)
data['文献综述'] = data['文献综述'].apply(convert_grade_to_score)
data['论文水平'] = data['论文水平'].apply(convert_grade_to_score)
data['论文写作'] = data['论文写作'].apply(convert_grade_to_score)

# 分离特征和目标变量
X = data[['论文选题', '文献综述', '论文水平', '论文写作']]
y = data['评分']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出模型性能
print('Mean Squared Error:', mse)
print('R2 Score:', r2)

# 输出回归系数
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

# 计算回归系数的绝对值总和
total = sum(abs(coefficients['Coefficient']))

# 计算每个系数的百分比
coefficients['Percentage'] = (abs(coefficients['Coefficient']) / total) * 100

print(coefficients)

# 设置 Matplotlib 支持中文的字体为 Arial Unicode MS
matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号无法显示的问题

# 自定义颜色
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']

# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(coefficients['Percentage'], labels=coefficients.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('论文指标对论文总分影响权重的占比')
# 保存饼图到本地文件
plt.savefig('regression_coefficients_pie_chart.png')
plt.show()

