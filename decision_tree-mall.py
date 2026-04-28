import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree 
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
X = df[['Age']]
df['Spending_Label'] = (df['Spending Score (1-100)'] > 50).astype(int)
y = df['Spending_Label']
model = DecisionTreeClassifier(max_depth=3)
print(model)
model.fit(X, y)
prediction=model.predict([[31]])
print("prection for age  :",prediction)
plt.figure(figsize=(12,8))
plot_tree(model,
           feature_names=["Age"],
          class_names=["Low","High"],
          filled=True,
          rounded=True,
          impurity=False)
plt.show(
)

