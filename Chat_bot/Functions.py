from re import X
from matplotlib.font_manager import json_load
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import json

stemmer = PorterStemmer()

with open ("contents.json","r") as f:
    file = json.load(f) 




def tokenize(sentence):
    return word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word, to_lowercase=True)


def bag_of_words(sentences):
    vectorizer = CountVectorizer()
    return vectorizer.fit_transform(sentences)



patterns=[]
for i in range(len(file["intents"])):
    patterns.extend(file['intents'][i]['patterns'])

X=bag_of_words(patterns)

