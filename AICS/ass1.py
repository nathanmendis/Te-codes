import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the dataset (cleaned version)
df = pd.read_csv("spam_dataset_3.csv", header=None, encoding='ISO-8859-1')
df = df.iloc[1:, [0, 1]]  # Remove the header row and keep only relevant columns
df.columns = ['label', 'message']
print(df.head())
# Vectorize the text messages
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['message'])
y = df['label']

# Train the Naive Bayes model
nb_model = MultinomialNB()
nb_model.fit(X, y)

# Predict a new message
new_message = "URGENT: Update your aadhar"
new_message_tfidf = tfidf_vectorizer.transform([new_message])
predicted_label = nb_model.predict(new_message_tfidf)

print("Predicted Label:", predicted_label[0])
