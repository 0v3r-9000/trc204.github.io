from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = load_iris(as_frame=True).frame

X = df.drop(columns='target')
y = df['target']

knc = KNeighborsClassifier()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

knc.fit(X_train,y_train)


knc.score(X_test,y_test)
knc.predict(X_test)
