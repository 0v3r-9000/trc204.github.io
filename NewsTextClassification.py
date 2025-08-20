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

----------------------------------------

# Example new text to classify
new_text = ["Gun politics in the United States is a deeply divisive issue, with significant partisan and demographic divides on issues like gun control and the role of firearms in society. Generally, Republicans tend to favor gun rights and ownership, while Democrats lean towards stricter gun control measures."]

# Use the trained pipeline to predict the categories of the new text
predictions = pipeline.predict(new_text)

# Print the predicted categories
for text, category_index in zip(new_text, predictions):
    predicted_category = newsgroups.target_names[category_index]
    print(f"Text: '{text}'\nPredicted Category: {predicted_category}\n")
