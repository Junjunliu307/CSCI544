# Python Version 3.11.4

import pandas as pd
import numpy as np
import nltk
import ssl
import re
from bs4 import BeautifulSoup

from sklearn.model_selection import train_test_split
import contractions

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# ssl._create_default_https_context = ssl._create_unverified_context
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
 
 # ! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Beauty_v1_00.tsv.gz

file_path = '../data/amazon_reviews_us_Office_Products_v1_00.tsv'

data = pd.read_csv(file_path, sep='\t', usecols=['review_body', 'star_rating'], low_memory=False)
data.rename(columns={'review_body': 'Review', 'star_rating': 'Rating'}, inplace=True)

RANDOM_NUM = 6
data['Rating'] = pd.to_numeric(data['Rating'], errors='coerce')

#drop NAN Value
data = data.dropna(subset=['Rating'])
data = data.dropna(subset=['Review'])

# print("Statistics of the ratings:")
# print(data['Rating'].value_counts().sort_values(ascending=False))

# Rating > 3 -> 1 positive，Rating <= 2 -> 0 negative，Rating == 3 -> None
data['Sentiment'] = data['Rating'].apply(lambda x: 1 if x > 3 else (0 if x <= 2 else None))

sentiment_counts = data['Sentiment'].value_counts(dropna=False)
print("Statistics of three classes (with comma between them)")
print("Positive Count: "+str(sentiment_counts.get(1, 0)) + ", Negative Count: "+ str(sentiment_counts.get(0, 0)) + ", Neutral Count: " + str(data['Sentiment'].isna().sum() ))

# drop neutral
data = data.dropna(subset=['Sentiment'])

# get positive && negative 10,000 comments
positive_reviews = data[data['Sentiment'] == 1].sample(10000, random_state=RANDOM_NUM)
negative_reviews = data[data['Sentiment'] == 0].sample(10000, random_state=RANDOM_NUM)

# print sample output
# print("Three sample reviews before data cleaning + preprocessing:")
# random_reviews = positive_reviews.sample(n=3, random_state=RANDOM_NUM)
# print(random_reviews[['Review', 'Rating']])

# print average len
# print(f"{positive_reviews['Review'].apply(len).mean():.2f} characters")
# print(f"{negative_reviews['Review'].apply(len).mean():.2f} characters")
avg_len_before_cleaning_pos = positive_reviews['Review'].apply(len).mean()
avg_len_before_cleaning_neg = negative_reviews['Review'].apply(len).mean()

# 1. lower case
positive_reviews['Review'] = positive_reviews['Review'].str.lower()
negative_reviews['Review'] = negative_reviews['Review'].str.lower()
# 2. remove HTML
positive_reviews['Review'] = positive_reviews['Review'].astype(str).apply(lambda x: BeautifulSoup(x, "html.parser").get_text() if "<" in x or ">" in x else x)
negative_reviews['Review'] = negative_reviews['Review'].astype(str).apply(lambda x: BeautifulSoup(x, "html.parser").get_text() if "<" in x or ">" in x else x)

# 3. remove URL
positive_reviews['Review'] = positive_reviews['Review'].apply(lambda x: re.sub(r"http\S+|www\S+", "", x))
negative_reviews['Review'] = negative_reviews['Review'].apply(lambda x: re.sub(r"http\S+|www\S+", "", x))

# 4. remove non-alphabetical
positive_reviews['Review'] = positive_reviews['Review'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))
negative_reviews['Review'] = negative_reviews['Review'].apply(lambda x: re.sub(r"[^a-zA-Z\s]", "", x))


# 5. remove extra spaces
positive_reviews['Review'] = positive_reviews['Review'].apply(lambda x: re.sub(r"\s+", " ", x).strip())
negative_reviews['Review'] = negative_reviews['Review'].apply(lambda x: re.sub(r"\s+", " ", x).strip())


# 6. perform contractions
positive_reviews['Review'] = positive_reviews['Review'].apply(contractions.fix)
negative_reviews['Review'] = negative_reviews['Review'].apply(contractions.fix)

avg_len_after_cleaning_pos = positive_reviews['Review'].apply(len).mean()
avg_len_after_cleaning_neg = negative_reviews['Review'].apply(len).mean()

print("Average length of reviews before and after data cleaning (with comma between them)")
print(str((avg_len_before_cleaning_neg+avg_len_before_cleaning_pos)/2.0)+", "+str((avg_len_after_cleaning_neg+avg_len_after_cleaning_pos)/2.0))
# print average len
# print(f"Average review length after cleaning (Positive): {positive_reviews['Review'].apply(len).mean():.2f} characters")
# print(f"Average review length after cleaning (Negative): {negative_reviews['Review'].apply(len).mean():.2f} characters")

# random_reviews = positive_reviews.sample(n=5, random_state=RANDOM_NUM)
# print(random_reviews[['Review', 'Rating']])

# print(f"Average review length before preprocessing (Positive): {positive_reviews['Review'].apply(len).mean():.2f} characters")
# print(f"Average review length before preprocessing (Negative): {negative_reviews['Review'].apply(len).mean():.2f} characters")
def remove_stop_words(review):
    # tokens = word_tokenize(review)
    tokens = review.split(" ")
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

positive_reviews['Review'] = positive_reviews['Review'].apply(remove_stop_words)

# random_reviews = positive_reviews.sample(n=5, random_state=RANDOM_NUM)
# print(random_reviews[['Review', 'Rating']])

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def perform_lemmatization(review):
    lemmatizer = WordNetLemmatizer()
    tokens = review.split(" ")
    pos_tags = nltk.pos_tag(tokens)
    
    lemmatized_tokens = []
    for word, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        lemmatized_tokens.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    
    return ' '.join(lemmatized_tokens)

positive_reviews['Review'] = positive_reviews['Review'].apply(perform_lemmatization)

# print(f"Average review length after preprocessing (Positive): {positive_reviews['Review'].apply(len).mean():.2f} characters")
# print(f"Average review length after preprocessing (Negative): {negative_reviews['Review'].apply(len).mean():.2f} characters")

# print("Three sample reviews after data cleaning + preprocessing:")
# random_reviews = positive_reviews.sample(n=3, random_state=RANDOM_NUM)
# print(random_reviews[['Review', 'Rating']])
avg_len_after_preprocessing_pos = positive_reviews['Review'].apply(len).mean()
avg_len_after_preprocessing_neg = negative_reviews['Review'].apply(len).mean()
print("Average length of reviews before and after data preprocessing (with comma between them)")
print(str((avg_len_after_cleaning_neg+avg_len_after_cleaning_pos)/2.0)+", "+str((avg_len_after_preprocessing_neg+avg_len_after_preprocessing_pos)/2.0))


all_reviews = pd.concat([positive_reviews, negative_reviews], ignore_index=True)

reviews = all_reviews['Review']
labels = all_reviews['Sentiment']

tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english') 

tfidf_features = tfidf_vectorizer.fit_transform(reviews)

tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

tfidf_df['Sentiment'] = labels.values

X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=RANDOM_NUM)
# print(X_train)
# print(y_train)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)

y_train_pred = perceptron.predict(X_train)

y_test_pred = perceptron.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
print("Accuracy, Precision, Recall, and f1-score for training and testing split (in the mentioned order) for Perceptron (with comma between them)")
print(f"Training: {train_accuracy:.4f}, {train_precision:.4f}, {train_recall:.4f}, {train_f1:.4f}")
print(f"Testing: {test_accuracy:.4f}, {test_precision:.4f}, {test_recall:.4f}, {test_f1:.4f}")


svm_model = SVC(kernel='linear', random_state=RANDOM_NUM)
svm_model.fit(X_train, y_train)

y_train_pred = svm_model.predict(X_train)

y_test_pred = svm_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Accuracy, Precision, Recall, and f1-score for training and testing split (in the mentioned order) for SVM")
print(f"Training: {train_accuracy:.4f}, {train_precision:.4f}, {train_recall:.4f}, {train_f1:.4f}")
print(f"Testing: {test_accuracy:.4f}, {test_precision:.4f}, {test_recall:.4f}, {test_f1:.4f}")

logistic_model = LogisticRegression(random_state=RANDOM_NUM)
logistic_model.fit(X_train, y_train)

y_train_pred = logistic_model.predict(X_train)

y_test_pred = logistic_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Accuracy, Precision, Recall, and f1-score for training and testing split (in the mentioned order) for Logistic Regression (with comma between them)")
print(f"Training: {train_accuracy:.4f}, {train_precision:.4f}, {train_recall:.4f}, {train_f1:.4f}")
print(f"Testing: {test_accuracy:.4f}, {test_precision:.4f}, {test_recall:.4f}, {test_f1:.4f}")



tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
tfidf_features = tfidf_vectorizer.fit_transform(reviews)

X_train, X_test, y_train, y_test = train_test_split(tfidf_features, labels, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_train_pred = nb_model.predict(X_train)
y_test_pred = nb_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Accuracy, Precision, Recall, and f1-score for training and testing split (in the mentioned order) for Naive Bayes (with comma between them)")
print(f"Training: {train_accuracy:.4f}, {train_precision:.4f}, {train_recall:.4f}, {train_f1:.4f}")
print(f"Testing: {test_accuracy:.4f}, {test_precision:.4f}, {test_recall:.4f}, {test_f1:.4f}")
