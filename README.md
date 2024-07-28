# Hate Speech Detection in Twitter

This project aims to identify and classify tweets into three categories: Hate Speech, Offensive Language, and No Hate or Offensive Language using various machine learning models. 

## Table of Contents
- [Introduction]
- [Dataset]
- [Installation]
- [Usage]
- [Models and Results]
- [Data Visualization]
- [Conclusion]
- [Contributing]
- [License]

## Introduction

With the rise of social media, the spread of hate speech and offensive language has become a significant issue. This project utilizes machine learning techniques to automatically detect and classify such content in tweets. Various models are compared to determine the most effective method for this task.

## Dataset

The dataset used for this project consists of tweets labeled as Hate Speech, Offensive Language, or No Hate or Offensive Language. It is stored in a CSV file named `twitter.csv`.

## Installation

To run this project locally, please ensure you have the following packages installed:

```python
import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
```

You can install these dependencies using pip:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn wordcloud
```

Additionally, download NLTK resources:

```python
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage

1. **Loading the Dataset:**

```python
df = pd.read_csv('twitter.csv')
print("Dataset loaded successfully.")
```

2. **Data Preprocessing:**

   - Convert text to lowercase
   - Remove URLs and punctuation
   - Remove stopwords

```python
def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    filtered_tweet = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_tweet)

df['tweet'] = df['tweet'].apply(preprocess_tweet)
```

3. **Feature Extraction:**

```python
vectorizer = CountVectorizer()
bow_features = vectorizer.fit_transform(df['tweet'])
```

4. **Model Training and Evaluation:**

   - Support Vector Machines (SVM)
   - Decision Tree Classifier
   - Random Forest Classifier
   - Naive Bayes
   - Logistic Regression

```python
# Example for SVM
X_train, X_test, y_train, y_test = train_test_split(bow_features, df['labels'], test_size=0.2, random_state=42)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

## Models and Results

| Model                   | Accuracy |
|-------------------------|----------|
| Support Vector Machines | 0.89     |
| Decision Tree Classifier| 0.88     |
| Random Forest Classifier| 0.88     |
| Naive Bayes             | 0.86     |
| Logistic Regression     | 0.90     |

## Data Visualization

Visualizations include the distribution of tweet categories and word clouds of the most frequent words in hate speech tweets.

```python
# Example: Word Cloud for Hate Speech tweets
hate_tweets = df[df.labels == "Hate Speech"]
text = ' '.join([word for word in hate_tweets['tweet']])
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```

## Conclusion

The project demonstrates the effectiveness of various machine learning models in detecting hate speech and offensive language on Twitter. Logistic Regression achieved the highest accuracy of 90%.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
