
import numpy as np
import pandas as pd


titanic_data = pd.read_csv("Titanic-Dataset.csv")

print(titanic_data.head())

titanic_data['Title'] = titanic_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1


titanic_data['IsAlone'] = 0
titanic_data.loc[titanic_data['FamilySize'] == 1, 'IsAlone'] = 1

titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())
titanic_data = titanic_data.drop(['PassengerId' ,'Pclass'], axis=1)
print("DATA AFTER PREPROCESSING")
print(titanic_data.head())