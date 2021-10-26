#################### Data Pre-processing: Natural language processing (NLP) for Sentiment Analysis ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/26
# Last Updated: 2021/10/26
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sentence_sentiment/nlpssentiment.py
#
#
########## Input Data File(s)
#
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python nlpssentiment.py 'The food was great!'
#python nlpssentiment.py "The food was great!"
#python nlpssentiment.py "The food was not great!"
#python nlpssentiment.py "The food was not that bad!"
#
#Generally,
#python nlpssentiment.py (one sentence)
#
#
########## Output Data File(s)
#
#
#
########## References
#
#Sentiment Analysis in Python: TextBlob vs Vader Sentiment vs Flair vs Building It From Scratch
#https://neptune.ai/blog/sentiment-analysis-python-textblob-vs-vader-vs-flair
#
#
#
####################


########## install Python libraries (before running this script)
#
#pip install textblob --upgrade
#pip install textblob --U
#python -m pip install textblob
#python -m pip install textblob==0.17.1
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade textblob --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade vaderSentiment --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
#pip install --upgrade torch --trusted-host pypi.org --trusted-host files.pythonhosted.org
#
#pip install --upgrade tiny-tokenizer --trusted-host pypi.org --trusted-host files.pythonhosted.org
#You'll see the following error;
#Exception: You tried to install "tiny_tokenizer". The name of tiny_tokenizer renamed to "konoha"
#
#pip install --upgrade konoha --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade flair --trusted-host pypi.org --trusted-host files.pythonhosted.org




########## import Python libraries

import sys




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #nlpssentiment.py

str_sentence = str(sys.argv[1])    #"The food was great!"




########## 1. Rule-based sentiment analysis

'''
The main drawback with the rule-based approach for sentiment analysis is that the method only cares about individual words and completely ignores the context in which it is used. 

For example, “the party was savage” will be negative when considered by any token-based algorithms.
'''

##### 1.1. Textblob
'''
It is a simple python library that offers API access to different NLP tasks such as sentiment analysis, spelling correction, etc.

Textblob sentiment analyzer returns two properties for a given input sentence: 

Polarity is a float that lies between [-1,1], -1 indicates negative sentiment and +1 indicates positive sentiments. 
Subjectivity is also a float which lies in the range of [0,1]. Subjective sentences with higher number like 1 generally refer to personal opinion, emotion, or judgment. 

Textblob will ignore the words that it doesn’t know, it will consider words and phrases that it can assign polarity to and averages to get the final score.
'''
from textblob import TextBlob

testimonial = TextBlob(str_sentence)
print(testimonial.sentiment)    #Sentiment(polarity=1.0, subjectivity=0.75)




##### 1.2. VADER sentiment

'''
Valence aware dictionary for sentiment reasoning (VADER) is another popular rule-based sentiment analyzer. 

It uses a list of lexical features (e.g. word) which are labeled as positive or negative according to their semantic orientation to calculate the text sentiment.   

Vader sentiment returns the probability of a given input sentence to be 

positive, negative, and neutral. 

Vader is optimized for social media data and can yield good results when used with data from twitter, facebook, etc.
'''

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

vs = analyzer.polarity_scores(str_sentence)
print("{:-<65} {}".format(str_sentence, str(vs)))    #{'compound': 0.6588, 'neg': 0.0, 'neu': 0.406, 'pos': 0.594}




########## 2. Embedding based models

'''
Text embeddings are a form of word representation in NLP in which synonymically similar words are represented using similar vectors which when represented in an n-dimensional space will be close to each other.

'''

##### 2.1. Flair 

from flair.models import TextClassifier
from flair.data import Sentence

classifier = TextClassifier.load('en-sentiment')
sentence = Sentence(str_sentence)
classifier.predict(sentence)

# print sentence with predicted labels
print('Sentence above is: ', sentence.labels)    #[POSITIVE (0.9961)
