import data_preprocessing as dp
import train_test_split as tts
import visualization as viz
import model_training as mt
import evaluation as ev

df = dp.load_data("/content/FuelConsumption.csv")

# Basic information about data
dp.data_info(df)

# Selecting specific columns for analysis
cdf = dp.select_columns(df)

# Split data into training and test sets
train, test = tts.split_data(cdf)

# Plotting data
viz.plot_data(train, test)

# Train the model
regr, train_x, train_y = mt.train_model(train)

# Output the coefficients
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

# Plotting the fitted line
viz.plot_fitted_line(train, train_x, train_y, regr)

# Predicting on test data
test_y_pred, test_y = mt.predict(regr, test)

# Evaluate the model
ev.evaluate(test_y_pred, test_y)
