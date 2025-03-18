import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

#Data Preprocessing
#load dataset
titanic_dataset = pd.read_csv("./Task1/Titanic-Dataset.csv")

#feature and dependent variable selection
y = titanic_dataset['Survived']
features= titanic_dataset.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


#handling missing data
missing_value=features.isnull().sum()
print('\nMissing data\n')
print(missing_value)

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

#train- test split
x_train,x_test,y_train,y_test= train_test_split(features,y, test_size=0.2,random_state=42, stratify=y)

#feature scaling using Standardization
sc = StandardScaler()
x_train.iloc[:,[6,-1]] = sc.fit_transform(x_train.iloc[:,[6,-1]])
x_test.iloc[:,[6,-1]] = sc.transform(x_test.iloc[:,[6,-1]])

#Model training 
titanic_model = LogisticRegression()
titanic_model.fit(x_train, y_train)

#model analysis
y_pred = titanic_model.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
print(f"The titanic model obtained an accuracy score of {accuracy*100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



