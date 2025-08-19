from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Load data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create pipeline
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=3)),
    ('classify', KNeighborsClassifier())
])

# Train and evaluate
pipeline.fit(X_train, y_train)
r2_score = pipeline.score(X_test, y_test)
print("Predictions are: ", pipeline.predict(X_test))
print("-------------------------------------------")
print("Score is: ", r2_score)
