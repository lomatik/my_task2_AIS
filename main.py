# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

testcsv = pd.read_csv('house-prices-advanced-regression-techniques/test.csv');
traincsv = pd.read_csv('house-prices-advanced-regression-techniques/train.csv');

print(traincsv)

print ("Train data shape:", traincsv.shape) # sizes of data
print ("Test data shape:", testcsv.shape)

data = traincsv.select_dtypes(include=[np.number]).interpolate().dropna() #interpolating data from train

y = np.log(traincsv.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
                          X, y, random_state=42, test_size=.33)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
print ("TRAIN_SCORE_R^2 is: \n", model.score(X_train, y_train))
print ("TEST_SCORE_R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)

print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.7,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

#Second part



