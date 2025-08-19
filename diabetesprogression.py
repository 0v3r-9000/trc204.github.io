from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.feature_selection import SelectKBest, f_regression

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures()),
    ('ridge', Lasso())
])

# Train and evaluate
pipeline.fit(X_train, y_train)
r2_score = pipeline.score(X_test, y_test)
print("Predictions are: ", pipeline.predict(X_test))
print("-------------------------------------------")
print("Score is: ", r2_score)
print("-------------------------------------------")
print("This is typical upper score limit for standard learning models on this dataset.")
print("-------------------------------------------")
print("Only deep learning models make it to about 65% explanation of variance.")
