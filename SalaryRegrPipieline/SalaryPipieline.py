from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns

df = sns.load_dataset('titanic')
df.head(10)
df.dtypes
numerical_features = ['age', 'fare']
categorical_features = ['embarked', 'sex', 'pclass']
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

preprocessor = ColumnTransformer([
    ('num_pipeline', numerical_pipeline, numerical_features),
    ('cat_pipeline', categorical_pipeline, categorical_features)
])
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])
X = df[['age', 'fare', 'embarked', 'sex', 'pclass']]
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
cv_score = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(cv_score)
print(np.mean(cv_score))
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['liblinear']
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

print("Best params: ", grid_search.best_params_)
print("Best CV acc: ", grid_search.best_score_)

-------------------------

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Predict on the test set using the best model from GridSearchCV
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Basic metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

-----------------------------------------

from sklearn.inspection import permutation_importance

result = permutation_importance(
    grid_search.best_estimator_,
    X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring='accuracy'
)

# Create a DataFrame
import pandas as pd
perm_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance Mean': result.importances_mean,
    'Importance Std': result.importances_std
}).sort_values(by='Importance Mean', ascending=False)

print(perm_df)\

print("----------------------------------------------")

import matplotlib.pyplot as plt

perm_df.plot(kind='barh', x='Feature', y='Importance Mean', legend=False)
plt.xlabel("Mean Importance")
plt.title("Permutation Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
