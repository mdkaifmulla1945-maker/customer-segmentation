import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
X = df[['Age']]
y = df['Spending Score (1-100)']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Linear Fit")
plt.legend()
plt.title("Linear Regression")
plt.show()