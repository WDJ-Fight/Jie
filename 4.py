import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import tree

# 导入训练数据
data = pd.read_csv(r"D:\Desktop\模式识别\titanic_train.csv")

# 数据清洗：
# 删除对分类无帮助的特征
data = data.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)

# 将某些字符串特征数值化: 将性别的值映射为数值
sex_mapDict = {'male': 1, 'female': 0}
data['Sex'] = data['Sex'].map(sex_mapDict)

# 均值填充缺失值
data.fillna(data.mean(),inplace=True)

# 简单建模
x = data.iloc[:, data.columns != 'Survived']
y = data.iloc[:, data.columns == 'Survived']
y = y.values.ravel()
rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre =  cross_val_score(rfc, x, y, cv=10).mean()
print(score_pre)

#调参n_estimators
scorel = []
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc, x, y, cv=10).mean()
    scorel.append(score)
print(max(scorel),(scorel.index(max(scorel))*10)+1)
best_n_estimators = (scorel.index(max(scorel))*10)+1
plt.figure(figsize=[20,5])
plt.plot(range(1,201,10),scorel)
plt.show()

#调整criterion
param_grid = {'criterion':['gini', 'entropy']}
rfc = RandomForestClassifier(n_estimators=best_n_estimators
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x,y)
print(GS.best_params_)
print(GS.best_score_)

#调整min_samples_split
param_grid={'min_samples_split':np.arange(2, 2+20, 4)}
rfc = RandomForestClassifier(n_estimators=best_n_estimators
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x,y)
print(GS.best_params_)
print(GS.best_score_)

#调整min_samples_leaf
param_grid={'min_samples_leaf':np.arange(1, 1+10, 1)}
rfc = RandomForestClassifier(n_estimators=best_n_estimators
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x,y)
print(GS.best_params_)
print(GS.best_score_)

#调参max_depth
param_grid = {'max_depth':np.arange(1, 10, 1)}

rfc = RandomForestClassifier(n_estimators=best_n_estimators
                             ,random_state=90
                            )
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(x, y)
print(GS.best_params_)
print(GS.best_score_)

#调参max_features
param_grid = {'max_features':np.arange(1,3,1)}
rfc = RandomForestClassifier(n_estimators=best_n_estimators
                             ,random_state=90
                            )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(x,y)
print(GS.best_params_)
print(GS.best_score_)

#RandomForest
clf = RandomForestClassifier(n_estimators = 71,
                            criterion = 'gini',
                            max_depth = 8,
                            min_samples_split = 2,
                            min_samples_leaf = 2,
                            max_features = 1,
                            random_state=90)
clf.fit(x, y)
print("RandomForest:",clf.score(x, y))

#Adaboost
clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=36)
clf.fit(x, y)
print(clf.feature_importances_)
print("Adaboost:",clf.score(x, y))

#DecisionTree
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i + 1
                                      , criterion="entropy"
                                      , random_state=90
                                      , splitter="random")
    clf = clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    test.append(score)
score = max(test)
print("DecisionTree:",score)