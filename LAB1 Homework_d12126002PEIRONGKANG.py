import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import re

# Function for cleaning data: assume that text data is located in a specific column.
def clean_text(text):
    # Clean up and deal with nonstandard characters or useless symbols
    text = re.sub(r'\t|\n', ' ', text)  # Remove tab and line breaks
    return text


file_paths = [
    r"D:\OneDrive\國立台灣大學\NTU_113-1\NTU_113-1\td_freq_db_alt_atheism.csv",
    r"D:\OneDrive\國立台灣大學\NTU_113-1\NTU_113-1\td_freq_db_comp_graphics.csv",
    r"D:\OneDrive\國立台灣大學\NTU_113-1\NTU_113-1\td_freq_db_sci_med.csv",
    r"D:\OneDrive\國立台灣大學\NTU_113-1\NTU_113-1\td_freq_db_soc_religion_christian.csv"
]


categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']


dataframes = []
for path, category in zip(file_paths, categories):
    df = pd.read_csv(path, header=None, names=['text'])
    df['text'] = df['text'].apply(clean_text)
    df['category'] = category
    dataframes.append(df)


data = pd.concat(dataframes, ignore_index=True)


print(data.head())

# Text vectorization
tfidf_vect = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
X_tfidf = tfidf_vect.fit_transform(data['text'])

# Divide data into training set and test set.
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['category'], test_size=0.3, random_state=42)

# Training Naive Bayesian Model
nb_tfidf = MultinomialNB()
nb_tfidf.fit(X_train, y_train)

# Make a prediction
y_pred_tfidf = nb_tfidf.predict(X_test)

print("Classification report (TF-IDF Vectorizer):")
print(classification_report(y_test, y_pred_tfidf))
print(f"Accuracy (TF-IDF Vectorizer): {accuracy_score(y_test, y_pred_tfidf)}")
