import numpy as np
from sklearn.model_selection import train_test_split

# Split data into training and test sets
def split_data(df):
    cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    train, test = train_test_split(cdf, test_size=0.2, random_state=42)
    return train, test
