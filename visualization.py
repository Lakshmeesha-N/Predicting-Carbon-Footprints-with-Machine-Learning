import matplotlib.pyplot as plt

def plot_data(train, test):
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue', marker='o')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.title("Train Data")
    plt.show()

  
    plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS, color='blue')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.title("Test Data")
    plt.show()

def plot_fitted_line(train, train_x, train_y, regr):
    plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
    plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
    plt.xlabel("Engine size")
    plt.ylabel("Emission")
    plt.title("Linear Fit")
    plt.show()
