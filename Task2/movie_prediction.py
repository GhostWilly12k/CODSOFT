import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

#Data preprocessing 
#load dataset
df = pd.read_csv("./Task2/IMDb-Movies-India.csv",encoding = "ISO-8859-1")

#handling missing values 
missing_values = df.isnull().sum()
print('\nMissing data')
print("_"*50)
print(missing_values)

df['Duration'] = pd.to_numeric(df['Duration'].str.extract(r'(\d+)')[0],errors='coerce')
df['Genre'] = df['Genre'].fillna('Unknown')
df['Votes'] = df['Votes'].str.extract(r'(\d+)')[0].astype(float)

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(df.iloc[:,[2,4,5]])
df.iloc[:,[2,4,5]] = imputer.transform(df.iloc[:,[2,4,5]])

df['Votes'] = df['Votes'].astype('int')
df['Duration'] = df['Duration'].astype('int')
df['Rating'] = df['Rating'].round(1)

df.dropna(inplace=True)

#data information 
print("\nDataset summary")
print("_"*50)
print(df.info())

print("\nDataset description")
print("_"*50)
print(df.describe())

#features and dependent variable selection 
# y = df['Rating']
# features = df.drop('Rating',axis = 1)

# print(y)
# print(features)
