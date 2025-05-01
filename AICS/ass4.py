
import numpy as np
import pandas as pd

# Load the Titanic dataset
titanic_data = pd.read_csv("train_dataset_4.csv")
# Display the first few rows of the dataset
print(titanic_data.head())

# Perform feature engineering

# 1. Extract titles from the 'Name' column
titanic_data['Title'] = titanic_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
# 2. Create a new feature 'FamilySize' by combining 'SibSp' and 'Parch' columns
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch'] + 1

# 3. Create a new feature 'IsAlone' indicating whether a passenger is alone or not
titanic_data['IsAlone'] = 0
titanic_data.loc[titanic_data['FamilySize'] == 1, 'IsAlone'] = 1
# 4. Convert 'Sex' feature into numerical
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

# 5. Convert 'Embarked' feature into numerical and fill missing values with the most common one
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# 6. Fill missing values in 'Age' column with median age
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].median())
# Display the modified dataset
print(titanic_data.head())