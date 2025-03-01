Titanic Survival Prediction Model
This was my first time building a prediction model, where I predicted whether a Titanic passenger survived. I carefully selected key features based on what I believed would impact survival: Pclass, Gender, Age, SibSp, Parch, and Fare.

Steps Taken:
Handled Missing Data: Used mean for numerical and mode for categorical values.
Encoded Categorical Data: One-hot encoding for Sex and Embarked.
Train-Test Split: 80% training, 20% testing (stratified).
Standardized Numerical Features: Applied to Age and Fare.
Model Training: Used Logistic Regression.
Results & Lessons Learned:
The model achieved 100% accuracy, which was unexpected and suggested overfitting or data leakage.
I learned the importance of cross-validation, feature selection, and trying different models for better generalization.
Next, I plan to explore Random Forest, feature engineering, and hyperparameter tuning.