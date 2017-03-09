# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer

documents = (
    "Macbook Pro 15' Silver Gray with Nvidia GPU",
    "Macbook GPU"
)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)[0,1])