import numpy as np


def evaluate(test_y_pred, test_y):
    mae = np.mean(np.absolute(test_y_pred - test_y))
    mse = np.mean((test_y_pred - test_y) ** 2)
    r2 = np.corrcoef(test_y_pred.T, test_y.T)[0, 1] ** 2

    print("Mean Absolute Error: %.2f" % mae)
    print("Mean Squared Error: %.2f" % mse)
    print("R2-score: %.2f" % r2)
