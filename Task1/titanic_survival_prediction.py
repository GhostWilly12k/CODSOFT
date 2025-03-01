import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#Data Preprocessing

#load dataset
titanic_dataset = pd.read_csv("./Task1/Titanic-Dataset.csv")

#feature and dependent variable selection
features= titanic_dataset.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
y = features['Survived']

#handling missing data
# missing_value=features.isnull().sum()
# print('\nMissing data\n')
# print(missing_value)

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(features.iloc[: ,3:4])
features.iloc[:,3:4] = imputer.transform(features.iloc[:,3:4])

imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer.fit(features.iloc[: ,[-1]])
features.iloc[:,[-1]] = imputer.transform(features.iloc[:,[-1]])

#encoding categorial data into numbers
encode_gender=pd.get_dummies(features['Sex']).astype(int)
features=pd.concat([encode_gender,features],axis=1)
features.drop(columns='female',axis=1,inplace=True)
features.drop(columns='Sex',axis=1,inplace=True)

encode_embarked=pd.get_dummies(features['Embarked']).astype(int)
features=pd.concat([encode_embarked,features],axis=1)
features.drop(columns='Embarked',axis=1,inplace=True)

print(y)