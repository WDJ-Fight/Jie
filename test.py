import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 导入训练数据
data = pd.read_csv(r"D:\Desktop\模式识别\实验四\titanic_train.csv")

# 数据清洗：
# 删除对分类无帮助的特征
data = data.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

# 将某些字符串特征数值化: 将性别的值映射为数值
sex_mapDict = {'male': 1, 'female': 0}
data['Sex'] = data['Sex'].map(sex_mapDict)

# 均值填充缺失值
data.fillna(data.mean(), inplace=True)

# 简单建模
x = data.iloc[:, data.columns != 'Survived']
y = data.iloc[:, data.columns == 'Survived']
y = y.values.ravel()
clf = RandomForestClassifier()
clf.fit(x,y)
print(clf.score(x,y))