# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from sklearn import linear_model

testcsv = pd.read_csv('house-prices-advanced-regression-techniques/test.csv');
sample_submissioncsv = pd.read_csv('house-prices-advanced-regression-techniques/sample_submission.csv');
traincsv = pd.read_csv('house-prices-advanced-regression-techniques/train.csv');

#print(sample_submissioncsv)
print(traincsv)

z = traincsv.loc[:, traincsv.columns != 'SalePrice']

X = np.array(z['MSSubClass'])#['LotArea']
X = X.reshape(1, -1)
y = np.dot(X, traincsv['SalePrice'])

reg = linear_model.LinearRegression().fit(X, y)

print('REG_PRINT')

print(reg.coef_)
print(reg.intercept_)

