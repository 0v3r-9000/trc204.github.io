import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')
df.head()
X = df['v2'].values
y = df['v1'].values
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
lb = LabelBinarizer()
y = lb.fit_transform(y).ravel()
lb.classes_
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X_trainTF = cv.fit_transform(X_train)
X_testTF = cv.transform(X_test)
print(X_testTF)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_trainTF, y_train)
y_pred = lr.predict(X_testTF)

for pred, v2 in zip(y_pred, X_test):
    print(pred, v2)
