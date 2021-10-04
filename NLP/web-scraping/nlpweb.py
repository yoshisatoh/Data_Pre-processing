#################### Data Pre-processing: Natural language processing (NLP) ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/04
# Last Updated: 2021/10/04
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/web-scraping/nlpweb.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/web-scraping/nlpweb.py
#
#
########## Input Data File(s)
#
#
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python nlpweb.py "https://en.wikipedia.org/wiki/SpaceX" 20
#
#Generally,
#python nlpweb.py (url) (num_words: number of words to show in descending order of counts for each word - if it's 20, then it shows top 20 words)
#
#
########## Output Data File(s)
#
#
#
########## References
#
#Gentle Start to Natural Language Processing using Python
#https://towardsdatascience.com/gentle-start-to-natural-language-processing-using-python-6e46c07addf3
#
#Natural Language Toolkit (NLTK)
#https://www.nltk.org/
#
#
####################




########## install Python libraries (before running this script)
#
#pip install nltk --upgrade
#pip install nltk --U
#python -m pip install nltk
#python -m pip install nltk==3.5
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade nltk --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade bs4 --trusted-host pypi.org --trusted-host files.pythonhosted.org
#pip install --upgrade html5lib --trusted-host pypi.org --trusted-host files.pythonhosted.org


########## import Python libraries

import sys

import nltk
#nltk.download()
from nltk.corpus import stopwords

import urllib.request
from bs4 import BeautifulSoup
import html5lib

import matplotlib.pyplot as plt

import csv

import pandas as pd




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #nlp_1_prep.py

url       = str(sys.argv[1])    #'https://en.wikipedia.org/wiki/SpaceX'

num_words = int(sys.argv[2])    #20




########## web page settings

response =  urllib.request.urlopen(url)
html = response.read()
#print(html)




########## getting text from the web page

soup = BeautifulSoup(html,'html5lib')
text = soup.get_text(strip = True)
#print(text)




########## tokens

tokens = [t for t in text.split()]

#print(tokens)
print(type(tokens))
print(len(tokens))




########## word frequency

sr= stopwords.words('english')

clean_tokens = tokens[:]


for token in tokens:
    if token in stopwords.words('english'):
        #
        clean_tokens.remove(token)
        #

freq = nltk.FreqDist(clean_tokens)
#print(freq)
#print(type(freq))
#<class 'nltk.probability.FreqDist'>


'''
for key,val in freq.items():
    #
    print(str(key) + ':' + str(val))
    #
'''


f = open('freq.csv' , 'w', encoding="utf-8")
writer = csv.writer(f, lineterminator='\n')
#
for key,val in freq.items():
    writer.writerow([str(key), str(val)])
#
#
f.close()
print(f.closed)



df = pd.read_csv('freq.csv', sep=',', header=None, index_col=False)
df = df.rename(columns={0: "key", 1: "val"})
print(df.head())
#print(df.values)
df_sorted = df.sort_values(by=['val'], ascending=False)
print(df_sorted.head())

df_sorted.to_csv('df_sorted.csv', index=False)  




########## word frequency: draw graphs


##### individual counts
fig = plt.figure(1, figsize = (10,8))
plt.gcf().subplots_adjust(bottom=0.20)    # to avoid x-ticks cut-off
freq.plot(num_words, cumulative=False, title='Top ' + str(num_words) + ' Most Common Words in Corpus: Individual')
plt.show()
fig.savefig('Fig_freq_individual.png', bbox_inches = "tight")
plt.close()


##### cumulative counts

fig = plt.figure(2, figsize = (10,8))
plt.gcf().subplots_adjust(bottom=0.20)    # to avoid x-ticks cut-off
freq.plot(num_words, cumulative=True, title='Top ' + str(num_words) + ' Most Common Words in Corpus: Cumulative')
plt.show()
fig.savefig('Fig_freq_cumulative.png', bbox_inches = "tight")
plt.close()

