import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path, sep=',')
    return df

def data_info(df):
    print("-------------------------\n BASIC INFORMATION ABOUT DATA\n------------------------------")
    print('DATA FRAME SIZE =', df.size)
    print("DATA FRAME SHAPE =", df.shape)
    print("NUMBER OF ELEMENTS IN EACH CLASS =\n", df.count())
    print(df.head(5))
    print(df.describe())

def select_columns(df):
    return df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
