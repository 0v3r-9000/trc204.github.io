from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = fetch_20newsgroups(subset='all')
categories = ['sci.med', 'sci.space', 'rec.sport.baseball', 'talk.politics.guns']
newsgroups = fetch_20newsgroups(
    subset='all',
    categories=categories,
    remove=('headers', 'footers', 'quotes'),
    shuffle=True,
    random_state=42
)

# Print dataset information
print(len(newsgroups.data))  # Number of documents
print(newsgroups.target_names)  # List of category names
print(newsgroups.target[:10])  # List of category codes

# Create a pipeline for text classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_df=0.5,
        min_df=2,
        stop_words='english',
        ngram_range=(1, 2)
    )),
    ('clf', SGDClassifier(
        loss='hinge',
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        learning_rate='optimal',
        penalty='l2'
    ))
])

# Split data into training and testing sets
X = newsgroups.data
y = newsgroups.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline
y_pred = pipeline.predict(X_test)

# Print evaluation metrics
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
