from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

ladf = pd.read_csv("loan_approval_dataset.csv")
ladf.head()

features = [' no_of_dependents', ' education', ' self_employed', ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value']
X = ladf[features]
y = ladf[' loan_status']
X_encoded = pd.get_dummies(X, columns=[' education', ' self_employed'], drop_first=True)

rf = RandomForestClassifier(random_state=42)
rf = rf.fit(X_encoded, y)
prediction = rf.predict(X_encoded)
print(accuracy_score(y, prediction))
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42)

# Display the shapes of the resulting sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)
# Initialize and train the Random Forest Classifier on the training data

rf.fit(X_train, y_train)

# Make predictions on the test data
predictions = rf.predict(X_test)

# Display the first few predictions
print("First 10 predictions on the test set:", predictions[:10])
# Calculate the accuracy of the model on the test data
accuracy = accuracy_score(y_test, predictions)

# Display the accuracy score
print("Accuracy of the Random Forest Model on the test set:", accuracy)
