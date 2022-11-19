from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(1)
df = pd.read_csv('pdata.csv')

rslt_df = df.drop(['Two_yr_Recidivism'],axis = 1)


output = df['Two_yr_Recidivism']

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(rslt_df, output, train_size=0.80)
list_train_pert = []
list_label = []
p = 0
for i in train.values:
    x = np.random.rand(train.shape[1])
    #print(x)
    list_train_pert.append(i+x)
    list_label.append(labels_train.values[p])
    p = p+1
#print(list_train_pert)
k = 0
for i in labels_train:
    if(train['African_American'].values[k] == 1):
        labels_train.values[k]= 1
    else:
        labels_train.values[k]= 0
    k = k+1
x = train.to_numpy()
y = labels_train.to_numpy()
train_x = np.concatenate((x, list_train_pert), axis = 0)

train_y = np.concatenate((y,list_label),axis = 0)
#print(train_x,train_y)
reg = LinearRegression().fit(train_x,train_y)
#y = reg.predict(test)
explainer = lime.lime_tabular.LimeTabularExplainer(train_x, feature_names=train.columns.values.tolist(),class_names=['Two_yr_Recidivism'], verbose=True, mode='regression')

j = 3
exp = explainer.explain_instance(test.values[j], reg.predict)
exp.save_to_file('lime5.html')