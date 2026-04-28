import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("Mall_Customers.csv")
print(df.head())
X = df[['Annual Income (k$)']] 
y = df['Spending Score (1-100)']
model = LogisticRegression()

model.fit(X ,y)
print("modeltrained")

pridict= model.predict(X)
print("pridict ",pridict)
probability = model.predict_proba([[5]])
print("probability",probability)
plt.scatter(X, y, label="Actual Data")
plt.plot(X, model.predict(X), label="Regression Line")

plt.xlabel(" annual income")
plt.ylabel("spending")
plt.legend()
plt.grid()
plt.title("Annual Income vs spending score Prediction")
plt.show()