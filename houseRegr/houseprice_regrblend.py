import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Load data
dftrain = pd.read_csv('train.csv')
dftest = pd.read_csv('test.csv')

# Feature engineering
for df in [dftrain, dftest]:
    df['TotalBaths'] = df['FullBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
    df['Has2ndFlr'] = (df['2ndFlrSF'] > 0).astype(int)

# Features
numeric_features = [
    'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',
    'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
    'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
    'TotalBaths', 'TotalSF', 'HasBsmt', 'Has2ndFlr'
]

categorical_features = ['Neighborhood']


X = dftrain[numeric_features]
X_test_kaggle = dftest[numeric_features]
y = np.log1p(dftrain['SalePrice'])

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# Pipeline for tuning
xgb_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('xgb', XGBRegressor(random_state=42, n_jobs=-1, verbosity=0))
])

# Search space
param_dist = {
    'xgb__n_estimators': [100, 300, 500, 800],
    'xgb__max_depth': [3, 4, 5, 6],
    'xgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.8, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8, 1.0]
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_pipe,
    param_distributions=param_dist,
    n_iter=25,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Fit search
random_search.fit(X_train, y_train)

# Best model
best_xgb = random_search.best_estimator_
print(f"✅ Best XGB CV Log RMSE: {-random_search.best_score_:.5f}")
print("Best params:", random_search.best_params_)

# Final preprocessing for stacking
X_train_proc = best_xgb.named_steps['preprocessor'].transform(X_train)
X_val_proc = best_xgb.named_steps['preprocessor'].transform(X_val)
X_test_proc = best_xgb.named_steps['preprocessor'].transform(X_test_kaggle)

# XGB predictions
xgb_train_pred = best_xgb.named_steps['xgb'].predict(X_train_proc).reshape(-1, 1)
xgb_val_pred = best_xgb.named_steps['xgb'].predict(X_val_proc).reshape(-1, 1)
xgb_test_pred = best_xgb.named_steps['xgb'].predict(X_test_proc).reshape(-1, 1)

# MLP meta-model
mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        max_iter=4000,
        early_stopping=True,
        random_state=42
    ))
])

# Fit MLP
mlp.fit(xgb_train_pred, y_train)

# Validate
mse = mean_squared_error(y_val, val_pred_log)
rmse = np.sqrt(mse)
print(f"📊 Final Tuned XGB → MLP Log RMSE: {rmse:.5f}")



# Predict on test set
final_log_preds = mlp.predict(xgb_test_pred)
final_preds = np.expm1(final_log_preds)

#/////////RESULTS/////////////

#Fitting 5 folds for each of 25 candidates, totalling 125 fits
# ✅ Best XGB CV Log RMSE: 0.13408
# Best params: {'xgb__subsample': 0.8, 'xgb__n_estimators': 800, 'xgb__max_depth': 5, 'xgb__learning_rate': 0.03, 'xgb__colsample_bytree': 0.7}
# 📊 Final Tuned XGB → MLP Log RMSE: 0.14445
# ✅ submission.csv created and ready for upload.

#/////////////////////////////

# Create submission
submission = pd.DataFrame({
    'Id': dftest['Id'],
    'SalePrice': final_preds
})
submission.to_csv('submission.csv', index=False)
print("✅ submission.csv created and ready for upload.")
