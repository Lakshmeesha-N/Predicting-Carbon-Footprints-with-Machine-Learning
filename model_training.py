from sklearn import linear_model
import numpy as np

# Train the model
def train_model(train):
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[['ENGINESIZE']])
    train_y = np.asanyarray(train[['CO2EMISSIONS']])
    regr.fit(train_x, train_y)
    return regr, train_x, train_y

# Predict on test data
def predict(regr, test):
    test_x = np.asanyarray(test[['ENGINESIZE']])
    test_y = np.asanyarray(test[['CO2EMISSIONS']])
    test_y_pred = regr.predict(test_x)
    return test_y_pred, test_y
