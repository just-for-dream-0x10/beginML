from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third document.",
    "Is this the first document?",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())

"""
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer


词性还原 + 词干提取
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("running", pos="v"))  # run
print(lemmatizer.lemmatize("better", pos="a"))   # good

stemmer = PorterStemmer()
print(stemmer.stem("running"))  # run
print(stemmer.stem("better"))   # better
"""

"""
文字规范化
import re
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

text = "Hello, World! 123 This is an example."
normalized_text = normalize_text(text)
print(normalized_text)  
"""
