import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

data = load_iris()
print(data)
# 划分数据集 8：2
x_train,x_test,y_train,y_test = train_test_split(data['data'],data['target'],test_size=0.2,random_state=22)
print(x_train.shape) # 120,4
print(y_train.shape) # 120,


#训练集可视化
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure()
c = ['r','b','g']   # 设置三个颜色
color = [c[y] for y in y_train] # 为不同的标签设置颜色，比如0--r--红色
plt.scatter(x_train[:,0],x_train[:,1],c=color)
plt.title('训练集图')
plt.xlabel('花萼长')
plt.ylabel('花萼宽')
plt.show()


# 创建模型
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
# 评估
score = model.score(x_test,y_test)
print('测试集准确率：',score)
# 评估2
y_predict = model.predict(x_test)
print('测试集对比真实值和预测值：',y_predict == y_test)


# 探究k值影响
score_list = []
for K in range (2,15):
    model_new = KNeighborsClassifier(n_neighbors=K)
    model_new.fit(x_train, y_train)
    score = model_new.score(x_test, y_test)
    score_list.append(score)

score_smoothed = gaussian_filter1d(score_list, sigma=1)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure()
plt.plot(range(2,15),score_smoothed)
plt.scatter(range(2,15),score_list,c="r")
plt.ylim(0.8,1)
plt.title('不同K值准确率')
plt.show()
print(score_list)



