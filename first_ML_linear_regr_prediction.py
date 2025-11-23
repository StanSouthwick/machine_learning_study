import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

df = pd.read_csv("https://content.codecademy.com/programs/data-science-path/linear_regression/honeyproduction.csv")

# Create linear regression object
regr = linear_model.LinearRegression()

# Take a look at the dataset
print(df.head())
prod_per_year = df.groupby('year').totalprod.mean().reset_index()

# Create the feature and target arrays
X = prod_per_year['year']
X = X.values.reshape(-1,1)

y = prod_per_year['totalprod']

# Train the linear regression model
regr.fit(X,y)
print(regr.coef_)
print(regr.intercept_)
y_predict = regr.predict(X)

# Plot the results
plt.scatter(X,y)
plt.plot(X,y_predict)
plt.show()
plt.close()

# Predict future honey production
X_future = np.array(range(2013,2051))
X_future = X_future.reshape(-1,1)
future_predict = regr.predict(X_future)

plt.scatter(X_future, future_predict)
plt.show()