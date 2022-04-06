import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    return(word_tokenize(sentence))

def stem(word):
    return stemmer.stem(word,to_lowercase=True)

def bag_of_words(tokenized_sentence,all_words):
    pass
