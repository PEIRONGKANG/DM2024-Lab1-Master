import os
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

dataset_path = "D:/OneDrive/國立台灣大學/NTU_113-1/NTU_113-1/20news-bydate.tar.gz"

if dataset_path.endswith("tar.gz"):
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall(path="20news-bydate")
def load_20newsgroups_data(path):
    data = []
    labels = []
    for root, dirs, files in os.walk(path):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='latin-1') as f:
                text = f.read()
                label = root.split(os.sep)[-1]
                data.append(text)
                labels.append(label)
    return pd.DataFrame({'text': data, 'category': labels})


train_data = load_20newsgroups_data("20news-bydate/20news-bydate-train")
test_data = load_20newsgroups_data("20news-bydate/20news-bydate-test")

print(f"Train Data Size: {train_data.shape}")
print(f"Test Data Size: {test_data.shape}")

count_vect = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)

#Vectorize training set and test set
X_train_counts = count_vect.fit_transform(train_data['text'])
X_test_counts = count_vect.transform(test_data['text'])
X_train_tfidf = tfidf_vect.fit_transform(train_data['text'])
X_test_tfidf = tfidf_vect.transform(test_data['text'])

nb_count = MultinomialNB()
nb_count.fit(X_train_counts, train_data['category'])

nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train_tfidf, train_data['category'])
y_pred_count = nb_count.predict(X_test_counts)
y_pred_tfidf = nb_tfidf.predict(X_test_tfidf)

print("Classification report (Count Vectorizer):")
print(classification_report(test_data['category'], y_pred_count))
print(f"Accuracy (Count Vectorizer): {accuracy_score(test_data['category'], y_pred_count)}")

# Output the evaluation results of the TF-IDF feature model
print("\nClassification report (TF-IDF Vectorizer):")
print(classification_report(test_data['category'], y_pred_tfidf))
print(f"Accuracy (TF-IDF Vectorizer): {accuracy_score(test_data['category'], y_pred_tfidf)}")
# Visualize the distribution of categories in the training set
train_data['category'].value_counts().plot(kind='bar', figsize=(10, 6), title='Category Distribution in Training Set')
plt.show()
