import numpy as np
import matplotlib.pyplot as mpl
import pandas as pd

# The location of the model's training set (just the dataset)
PATH = "training_set.csv"

data = pd.read_csv(PATH)

# Function formats and loads the data needed for the regression analysis


def load_data():
    data = pd.read_csv(PATH)
    m = len(data)
    x = np.array(data[data.columns[1]]).reshape((m, 1))
    y = np.array(data[data.columns[2]]).reshape((m, 1))
    return [x, y, m]
