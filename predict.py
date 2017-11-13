import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# I have a small dataset of features of 3 types of flowers. [Sepal length,Sepal width,Petal length,Petal width,Species]
df = pd.read_csv('Iris.csv')
df.head(10)