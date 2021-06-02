import string
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

class Preprocessor:
    
    def __init__(self,df,features,fillna = False):
        self.df = df
        self.fillna = fillna
        self.features = features


    def preprocess_text(self,text):

        clean_text = text.strip()
        clean_text = clean_text.lower()
        clean_text = re.sub(r'\d+','',clean_text)
        lookup_table = clean_text.maketrans('', '', string.punctuation)
        clean_text = clean_text.translate(lookup_table)
        word_list = word_tokenize(clean_text)
        word_list = [w for w in word_list if not w in stop_words]
        word_list = [lemmatizer.lemmatize(word) for word in word_list]
        clean_text = ' '.join(word_list)

        return clean_text


    def preprocess(self):

        for feature in self.features:
            if len(self.df[self.df[feature].map(type)!=str]) == 0:
                self.df[feature] = self.df[feature].map(self.preprocess_text)
