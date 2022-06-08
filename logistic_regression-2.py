# Importing the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Datasets/advertising.csv')

# Understanding the Data
'''sns.distplot(df['Age'], kde=False, bins=30)
sns.countplot(x='Male', hue='Clicked on Ad', data=df)
sns.countplot(x='Age', hue='Clicked on Ad', data=df)'''

# Cleaning the Data
df = df.drop(['Timestamp', 'Country', 'City', 'Ad Topic Line'], axis=1)
print(df.head())

# Predict the Data
X = df.drop('Clicked on Ad', axis=1)
y = df['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

# Evaluating the results
plt.scatter(y_test, predictions,)
plt.show()
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
