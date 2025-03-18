import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load Dataset
df = pd.read_csv("./Task5/creditcard.csv")

# Data description 
print("\nDataset description")
print("_"*50)
print(df.describe())

# Data summary 
print("\nDataset summary")
print("_"*50)
print(df.info())

# Handling missing data
missing_value=df.isnull().sum()
print('\nMissing data')
print("_"*50)
print(missing_value)
print("No missing data found")

# Duplicate handling
print('\nDuplicated data')
print("_"*50)
print(f"{df.duplicated().sum()} found , all dropped")
df.drop_duplicates(inplace = True)

# Exploratory Data Analysis
# Volumes of Transactions over the two days
day1 = (df["Time"] < 86400).sum()   
day2 = (df["Time"] >= 86400).sum() 
plt.figure(figsize=(6, 4))
plt.bar(["Day 1", "Day 2"], [day1, day2], color=["blue", "orange"])
plt.xlabel("Day")
plt.ylabel("Number of Transactions")
plt.title("Transactions Count Over 2 Days")
plt.show()

# Volume of fradulent transactions over the two days
day1_f = ((df["Time"] < 86400) & (df["Class"] == 1)).sum()   
day2_f = ((df["Time"] >= 86400) & (df["Class"] == 1)).sum() 
plt.figure(figsize=(6, 4))
plt.bar(["Day 1", "Day 2"], [day1_f, day2_f], color=["blue", "orange"])
plt.xlabel("Day")
plt.ylabel("Number of Fraudulent Transactions")
plt.title("Fraudulent Transactions Count Over 2 Days")
plt.show()
print(f"Day 1 had {day1_f} fradulent transactions and Day 2 had {day2_f} fradulent transactions")

# True transactions vs Fradulent transactions
print(f"Total transactions made over the two days are equal to {day1+day2}")
print(f"Only {(day1_f+day2_f)/(day1+day2):.5f}% of total transactions where fradulent transactions , this shows the imbalance between fradulent and non fradulent transactions ")

# Feature and Target split 
features = df.drop(columns=['Time','Class',],axis=1)
y = df["Class"]

# Train-Test split
x_train,x_test,y_train,y_test= train_test_split(features,y, test_size=0.2,random_state=42, stratify=y)

# Feature scaling using Standardization
sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model 1 - Without Imbalance Handling
rf_model1 = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model1.fit(x_train, y_train)
y_pred = rf_model1.predict(x_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel 1 - Without Imbalance Handling")
print("_"*50)
print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1 Score: {f1:.5f}")

y_pred_proba = rf_model1.predict_proba(x_test)[:, 1]
auprc = average_precision_score(y_test, y_pred_proba)
print(f"AUPRC: {auprc:.5f}")
print("\n")

# Model 2 - With SMOTE Imbalance Handling and Undersampling using RandomUnderSampler"
rf_model2 = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
smote = SMOTE(random_state=42)
x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)
rus =RandomUnderSampler(sampling_strategy=1.0,random_state=42)
x_train_sa,y_train_sa=rus.fit_resample(x_train_sm,y_train_sm)
rf_model2.fit(x_train_sa, y_train_sa)
y_pred = rf_model2.predict(x_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel 2 - With SMOTE Imbalance Handling and Undersampling using RandomUnderSampler")
print("_"*50)
print(f"Precision: {precision:.5f}")
print(f"Recall: {recall:.5f}")
print(f"F1 Score: {f1:.5f}")

y_pred_proba = rf_model2.predict_proba(x_test)[:, 1]
auprc = average_precision_score(y_test, y_pred_proba)
print(f"AUPRC: {auprc:.5f}")

