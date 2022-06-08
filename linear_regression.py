# import data manipulating libraries
import pandas as pd
import numpy as np

# import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# import machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# getting the pokemon data set and modifying it
df = pd.read_csv('Datasets/seaborn-data-master/mpg.csv')
df = df.dropna()
df = df.drop(['name', 'origin', 'model_year'], axis=1)

# initializing the features and the target variable
X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']]
y = df['mpg']

# train_test_split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# create a linear regression model and training it
lm = LinearRegression()
lm.fit(X_train, y_train)

# printing the coefficients for the data
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficients'])
print(cdf)

# finding the predictions for the data
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.show()

# creating a histogram of the residuals
sns.distplot((y_test-predictions))
plt.show()

# printing the different types of errors
mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
print(mae)
print(mse)
print(rmse)
