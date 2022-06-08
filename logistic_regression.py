# Importing the needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Collecting the dataset
train = pd.read_csv('Datasets/titanic_train.csv')

# Visualizing and Understanding the data
sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=train)

sns.distplot(train['Age'].dropna(), kde=False, bins=30)

sns.countplot(x='SibSp', data=train)

plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)


# Cleaning the data
def impute_age(cols):
    age = cols[0]
    pclass = cols[1]

    if pd.isnull(age):
        if pclass == 1:
            return 37
        elif pclass == 2:
            return 29
        else:
            return 24
    else:
        return age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
gender = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, gender, embark], axis=1)
train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Machine Learning !!!
X = train.drop('Survived', axis=1)
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

print(confusion_matrix(y_test, predictions))
