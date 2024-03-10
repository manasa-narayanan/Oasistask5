import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv("avd.csv")
print(data.head())
print(data.isnull().sum())
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr())
plt.show()
