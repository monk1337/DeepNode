import numpy as np
import pandas as pd
import re,string
from bs4 import BeautifulSoup

from sklearn.preprocessing import LabelBinarizer

class Text_Preprocessing(object):
    
    def __init__(self, dataset, trim_length, le = False):
        self.dataset         = dataset
        self.trim_           = trim_length
        self.le              = le

    #Removing the html strips
    def strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    #Removing the square brackets
    def remove_between_square_brackets(self,text):
        return re.sub('\[[^]]*\]', '', text)

    #Removing the noisy text
    def denoise_text(self,text):
        text = self.strip_html(text)
        text = self.remove_between_square_brackets(text)
        return text.lower().strip()


    #Define function for removing special characters
    def remove_special_characters(self, text, remove_digits=True):
        pattern=r'[^a-zA-z0-9\s]'
        text=re.sub(pattern,'',text)
        return text

    # trim the sentences to average length 
    def trim_sentences(self, sentence):
        return " ".join(sentence.split()[:self.trim_])
    
    
    def label_encoding(self, dataset):
        lb             = LabelBinarizer()
        sentiment_data = lb.fit_transform(dataset['sentiment'])
        labels = np.array(sentiment_data).squeeze()
        dataset['encoded_labels'] = labels
        return dataset
    
    # all steps
    def preprocessing(self):
        
        pandas_frame           = pd.read_csv(self.dataset)
        pandas_frame['review'] = pandas_frame['review'].apply(self.denoise_text)
        pandas_frame['review'] = pandas_frame['review'].apply(self.remove_special_characters)
        pandas_frame['review'] = pandas_frame['review'].apply(self.trim_sentences)
        
        if self.le:
            pandas_frame = self.label_encoding(pandas_frame)
            
        
        return pandas_frame