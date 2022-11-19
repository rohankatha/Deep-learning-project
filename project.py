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
print(train.values[2])
reg = LinearRegression().fit(train,labels_train)
#y = reg.predict(test)
explainer = lime.lime_tabular.LimeTabularExplainer(train.values, feature_names=train.columns.values.tolist(),class_names=['Two_yr_Recidivism'], verbose=True, mode='regression')

j = 5
exp = explainer.explain_instance(test.values[j], reg.predict,num_features=6)
exp.save_to_file('lime5.html')