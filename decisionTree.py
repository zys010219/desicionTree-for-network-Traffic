#导出决策树结构图后在命令行中在该目录下执行命令dot tree1.dot -Tpng -o tree1.png以转换为png图。

#导入相关库
import sklearn
import sklearn.model_selection as sk_model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree                                 #导入tree模块
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import ADASYN
from collections import Counter
import pandas as pd
import graphviz
import numpy as np
import time

netTraffic_data = pd.read_csv('src-train.csv')  #dataframe格式
netTraffic_test = pd.read_csv('src-test.csv')

#print(netTraffic_test.head())
#print(netTraffic_test.info())

#scaler = MinMaxScaler(feature_range=[0,1]) #归一化（不需）
#rus = ADASYN(random_state=42)              #欠采样/过采样 效果差于集成采样


test_x = netTraffic_test[['Q1_IAT', 'Med_IAT', 'Min_IAT',
                          'Q3_IAT', 'Max_IAT', 'Var_IAT', 'Mean_IAT']]
test_y = []
vec = DictVectorizer(sparse=False)
test_X = vec.fit_transform(test_x.to_dict(orient='record'))

#print(test_X)
#print(vec.feature_names_)


#print(netTraffic_data.shape)
#print(netTraffic_data.head())
#print(netTraffic_data.info())


train_x = netTraffic_data[['Q1_IAT', 'Med_IAT','Min_IAT',
                           'Q3_IAT', 'Max_IAT', 'Var_IAT', 'Mean_IAT']]
train_y = netTraffic_data['Classes']
print("各类样本数量：", sorted(Counter(train_y).items()))
#print(train_x.head())
#print(train_y.head())
#resampleX,resampleY = rus.fit_resample(train_x,train_y)
#print(resampleX)
#print(resampleY)

#print(sorted(Counter(resampleY).items()))


#使用feature_extraction中的特征转换器，把类别变量中的特征都单独剥离出来，独立成一列特征
vec = DictVectorizer(sparse=False)
train_X = vec.fit_transform(train_x.to_dict(orient='record'))
#print("训练数据"+train_X)
print("用于构建决策树的参数：", vec.feature_names_)

#scalerX = scaler.fit_transform(train_X)
#scalerTest = scaler.fit_transform(test_X)
#print(scalerX)
#print(scalerTest)

#采用C4.5算法进行计算
#获取模型
t1 = time.time()
modeltree = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=None, min_samples_split=9,
                                    min_samples_leaf=5, min_weight_fraction_leaf=0.0, max_features=None,
                                    random_state=None, max_leaf_nodes=None, class_weight=None)
modeltree.fit(train_X, train_y)

'''
#GridSearchCV函数寻求最优参数
tree_param_grid = {'max_depth': list((3,5,7,9,11,13)),'min_samples_split':list((3,6,9,12)),'min_samples_leaf':list((3,5,7,9)), 'max_features':list((1,2,3,5))}
grid = GridSearchCV(modeltree,param_grid=tree_param_grid, cv=10)
grid.fit(train_X, train_y)
print(grid.best_params_)
'''

model = BaggingClassifier(base_estimator=modeltree, random_state=42) #集成

model.fit(train_X, train_y)
t2 = time.time()
print("模型构建用时：", t2-t1, "s")
y_predict = model.predict(test_X)
print("分类结果：", y_predict)

feature_name = ['Q1_IAT', 'Med_IAT', 'Min_IAT',
                          'Q3_IAT', 'Max_IAT', 'Var_IAT', 'Mean_IAT']

dot_data = tree.export_graphviz(modeltree
                                , out_file=None
                                , feature_names=feature_name
                                , class_names=['WWW', 'MAIL', 'P2P', 'FTP-DATA']
                                , filled=True
                                , rounded=True)
graph = graphviz.Source(dot_data)
graph.save('tree1.dot')
print("已导出决策树结构图")

accs=sk_model_selection.cross_val_score(model, train_X, y=train_y, scoring=None,cv=10, n_jobs=1)
print('交叉验证结果:',accs)

for i in range(len(y_predict)):
    if y_predict[i] == 'WWW':
        y_predict[i] = 0
    elif y_predict[i] == 'MAIL':
        y_predict[i] = 1
    elif y_predict[i] == 'P2P':
        y_predict[i] = 2
    elif y_predict[i] == 'FTP-DATA':
        y_predict[i] = 3


print("结果映射为数字：", y_predict)
result = pd.DataFrame(y_predict)


print(result)
result.to_csv('result.csv')
print("已导出结果至result.csv")
