import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Data preprocessing 
# Load Dataset
df = pd.read_csv("./Task2/IMDb-Movies-India.csv",encoding = "ISO-8859-1")

# Handling missing values 
missing_values = df.isnull().sum()
print('\nMissing data')
print("_"*50)
print(missing_values)

df['Duration'] = pd.to_numeric(df['Duration'].str.extract(r'(\d+)')[0],errors='coerce')
df['Genre'] = df['Genre'].fillna('Unknown')
df['Votes'] = df['Votes'].str.extract(r'(\d+)')[0].astype(float)

# Impute missing values with median
imputer = SimpleImputer(missing_values=np.nan,strategy='median')
imputer.fit(df.iloc[:,[2,4,5]])
df.iloc[:,[2,4,5]] = imputer.transform(df.iloc[:,[2,4,5]])

# Data type conversion
df['Votes'] = df['Votes'].astype('int')
df['Duration'] = df['Duration'].astype('int')
df['Rating'] = df['Rating'].round(1)

# Drop rows with missing values
df.dropna(inplace=True)

df['Year'] = df['Year'].str.extract(r'(\d+)').astype(int) #fixing year format

#duplicate handling
print('\nDuplicated data')
print("_"*50)
print(df.duplicated().sum())
df.drop_duplicates(inplace = True)

#data information 
print("\nDataset summary")
print("_"*50)
print(df.info())

print("\nDataset description")
print("_"*50)
print(df.describe())

# Exploratory Data Analysis 
# Year with the best movie rating 
avg_rating_per_year = df.groupby('Year')['Rating'].mean()
best_year = avg_rating_per_year.idxmax()
best_rating = avg_rating_per_year.max()
print(f"\nThe year with the best movie rating is {best_year} with an average rating of {best_rating:.1f}")

# Average ratings per year
sns.set_theme(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_rating_per_year.index, y=avg_rating_per_year.values,marker='o',color='b',linewidth=2.5,markersize=8)
plt.xlabel("Year", fontsize=14, fontweight='bold')
plt.ylabel("Average Rating", fontsize=14, fontweight='bold')
plt.title("Movie Ratings Trend Over Time", fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Director with the most movies
num_movies_by_dir = df.groupby('Director')['Name'].count()
director = num_movies_by_dir.idxmax()
num_movies = num_movies_by_dir.max()
print(f"{director} is the movie director who directed the most movies, leading with over {num_movies} movies")

# Feature engineering : calculate director and actor ratings 
cols = ['Director','Actor 1', 'Actor 2', 'Actor 3']
for col in cols:
    actor_ratings = df.groupby(col)['Rating'].mean().reset_index()
    actor_ratings = actor_ratings.rename(columns={'Rating': f'{col}_rating'})
    df = pd.merge(df, actor_ratings, on=col, how='left').drop(col,axis =1)
    df[f'{col}_rating'] = df[f'{col}_rating'].round(1)
    
# Feature engineering : Genre encoding with genre correlation > threshold
df_genres = df['Genre'].str.get_dummies(",")
correlation = df_genres.corrwith(df['Rating'].sort_values(ascending=False))
threshold = 0.06
genres = correlation[correlation.abs()> threshold]
df = df.join(df_genres[genres.index]).drop(columns=['Genre'])

# Feature correlation
# Duration vs Rating
correlation = df['Duration'].corr(df['Rating'])
print(f"\nThe correlation between the duration of the movie and its rating is {correlation:.4f}.\n"
      f"This indicates a very weak relationship between the movie's length and how it is rated.\n"
      f"Essentially, the duration of the movie has little to no influence on the ratings provided by critics or viewers.")

# Director ratings vs Movie ratings
correlation = df['Director_rating'].corr(df['Rating'])
print(f"\nThe correlation between the director's rating and the movie's rating is {correlation:.4f}.\n"
      f"This indicates a strong positive relationship between the director's rating and how the movie is rated.\n"
      f"Essentially, movies directed by highly rated directors tend to receive higher ratings from critics and viewers.")

# Correlation of Rating vs Other features
corr_df = df.drop(columns=['Name'])
rating_correlation = corr_df.corr()['Rating'].sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=rating_correlation.index, y=rating_correlation.values, palette="coolwarm")
plt.title('Correlation of Rating with Other Features', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=14, fontweight='bold')
plt.ylabel('Correlation', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()
plt.show()

print("""
After evaluating the correlation heatmap on the movie features , I discovered that there is a strong correlation 
between director and actor ratings and the overall movie ratings. 
In contrast, the movie genre showed a very low correlation with the ratings. 
Additionally, I found that the duration of the films had no correlation 
with the ratings at all.

As a result of these findings, I decided to select features with a correlation 
greater than 0.05 for my machine learning model.
""")

# Features and Target variable selection 
y = df['Rating']
features = df.drop(columns=['Rating','Name','Duration'],axis = 1)

# Train-test split 
x_train,x_test,y_train,y_test= train_test_split(features,y, test_size=0.2,random_state=42)

# Feature scaling using Standardization
sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Model training 
model = LinearRegression()
model.fit(x_train,y_train)

# Model Predictions
y_pred = model.predict(x_test)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"The Mean Squared Error (MSE) of the model is {mse:.2f}, which tells us that, "
      f"on average, the model's predictions are fairly close to the actual movie ratings.\n"
      f"The Root Mean Squared Error (RMSE) comes in at {rmse:.2f}, meaning the predicted "
      f"ratings are off by just under {rmse:.2f} points from the actual ratings—indicating "
      f"relatively small errors.\nWith an R-squared (R²) value of {r2:.2f}, the model is able "
      f"to explain about {r2 * 100:.2f}% of the variation in movie ratings.\nThis shows that "
      f"the model captures most of the factors influencing ratings, though there's still some "
      f"room for improvement.\n\nAll in all, these metrics suggest the model is doing a solid "
      f"job, but there's potential to refine it further for even better performance.")

# Predicted vs Actual 
plt.figure(figsize=(8, 6))  # Set the figure size for better visibility
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predictions')  # Scatter plot with a little transparency
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2, label='Ideal Line')  # Ideal line
plt.title('Predicted vs Actual Movie Ratings', fontsize=16)
plt.xlabel('Actual Ratings', fontsize=14)
plt.ylabel('Predicted Ratings', fontsize=14)
plt.legend(loc='upper left', fontsize=12)  # Add a legend
plt.grid(True)  # Add gridlines for easier interpretation
plt.show()