#################### Data Pre-processing: Natural language processing (NLP) for 10-k filings ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/05
# Last Updated: 2021/10/05
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sec-edgar_nlp/sec-edgar_nlp.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/sec-edgar_nlp/sec-edgar_nlp.py
#
#
########## Input Data File(s)
#
#browse-edgar.txt
#CIK_ticker.csv
#loughran_mcdonald_master_dic_2016.csv
#yr-quotemedia.csv
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#python sec-edgar_nlp.py 0001018724 10-K
#
#Generally,
#python sec-edgar_nlp.py (cik) (doc_type)
#
#
########## Output Data File(s)
#
#browse-edgar.xml
#list_raw_fillings_by_ticker_url.txt
#list_raw_fillings_by_ticker_url_file.txt
#
#
########## References
#
#NLP in the Stock Market
#https://towardsdatascience.com/nlp-in-the-stock-market-8760d062eb92
#https://github.com/roshan-adusumilli/nlp_10-ks
#https://github.com/roshan-adusumilli/nlp_10-ks/blob/master/NLP_on_Financial_Statements.ipynb
#https://github.com/roshan-adusumilli/nlp_10-ks/blob/master/project_helper.py
#https://github.com/roshan-adusumilli/nlp_10-ks/blob/master/project_tests.py
#https://github.com/roshan-adusumilli/nlp_10-ks/blob/master/tests.py
#
#loughran_mcdonald_master_dic_2016.csv
#https://github.com/soheil-mpg/NLP-on-Financial-Statements/blob/master/loughran_mcdonald_master_dic_2016.csv
#
#yr-quotemedia.csv
#https://github.com/soheil-mpg/NLP-on-Financial-Statements/blob/master/yr-quotemedia.csv
#(Or, access: https://www.quotemedia.com/contentsolutions/historical_data)
#
#Natural Language Toolkit (NLTK)
#https://www.nltk.org/
#
#
####################




########## Background
#
#SEC EDGAR database provides various forms of filings.
#
#10-K forms are annual reports filed by companies to provide a comprehensive summary of their financial performance (these reports are mandated by the Securities and Exchange Commission). 
#Combing through these reports is often tedious for investors.
#Through sentiment analysis, a subfield of natural language processing, investors can quickly understand if the tone of the report is positive, negative, or litigious etc.
#The overall sentiment expressed in the 10-k form can then be used to help investors decide if they should invest in the company.
#
#10-Q forms are QUARTERLY reports.




########## Overview of NLP

'''
Natural language processing is a branch of artificial intelligence concerned with teaching computers to read and derive meaning from language.
Since language is so complex, computers have to be taken through a series of steps before they can comprehend text.
The following is a quick explanation of the steps that appear in a typical NLP pipeline.

1. Sentence Segmentation
The text document is segmented into individual sentences.

2. Tokenization (Word Segmentation as a pre-processing)
Once the document is broken into sentences, we further split the sentences into individual words. Each word is called a token, hence the name tokenization.

3. Parts-of-Speech-Tagging
We input each token as well as a few words around it into a pre trained part-of-speech classification model to receive the part-of-speech for the token as an output.

4. Lemmatization
Words often appear in different forms while referring to the same object/action.
To prevent the computer from thinking of different forms of a word as different words, we perform lemmatization, the process of grouping together various inflections of a word to analyze them as a single item, identified by the word’s lemma (how the word appears in the dictionary).

5. Stop Words
Extremely common words such as “and”, “the” and “a” don’t provide any value, so we identify them as stop words to exclude them from any analysis performed on the text.

6. Dependency Parsing
Assign a syntactic structure to sentences and make sense of how the words in the sentence relate to each other by feeding the words to a dependency parser.

7. Noun Phrases
Grouping the noun phrases in a sentence together can help simplify sentences for cases when we don’t care about adjectives.

8. Named Entity Recognition (NEP)
A Named Entity Recognition model can tag objects such as people’s names, company names, and geographic locations.

9. Coreference Resolution
Since NLP models analyze individual sentences, they become confused by pronouns referring to nouns from other sentences.
To solve this problem, we employ coreference resolution which tracks pronouns across sentences to avoid confusion.

After going through these steps of data-preprocessing, our text is ready for analysis.

(Please refer to the original page at towardsdatascience.com for more details.)
'''




########## install Python libraries (before running this script)
#
#pip install nltk --upgrade
#pip install nltk --U
#python -m pip install nltk
#python -m pip install nltk==3.5
#
#If any of the above does not work in your environment, then try:
#pip install --upgrade nltk --trusted-host pypi.org --trusted-host files.pythonhosted.org




########## import Python libraries

import os
import sys
import csv
import time

import alphalens as al
import ratelimit
import requests
import sklearn
import six

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import pprint
#import tqdm
from tqdm import tqdm

from bs4 import BeautifulSoup
#import lxml

#You may have to run the following for the first time.
#
#nltk.download('stopwords')
#nltk.download('wordnet')


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords


from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer


#from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score


from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.metrics.pairwise import cosine_similarity


import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import warnings
warnings.simplefilter('ignore')


#Reference:
#https://github.com/roshan-adusumilli/nlp_10-ks
import project_helper
import project_tests
import tests




########## argument(s)

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #nlpweb10k.py

#SEC EDGAR database
arg_cik      = str(sys.argv[1])    #0001018724
arg_doc_type = str(sys.argv[2])    #10-K




########## pre-defined parameter(s)

#SEC EDGAR database
start    = 0     # 0 means "from the latest one" such as the latest year document in the case of 10-K annual reports.
count    = 100   # Limit Results Per Page on the 
t        = 15    #time.sleep(t) to let users download an xml file from the SEC EDGAR website before running the rest of this script




########## Get a 10-K xml file from the SEC EDGAR database website
'''
10-k documents include information such as company history, organizational structure, executive compensation, equity, subsidiaries, and audited financial statements.
To lookup 10-k documents, we use each company’s unique CIK (Central Index Key).

For instance, if you look at AMAZON COM INC,
its "ticker" is 'AMZN' (on Nasdaq) while its CIK is '0001018724'


CIK (Central Index Key) lookup:
https://www.sec.gov/edgar/searchedgar/cik.htm

If you want to see the detail of a certain company with CIK '0001018724' then access:
https://www.sec.gov/edgar/browse/?CIK=1018724
or
https://www.sec.gov/edgar/browse/?CIK=0001018724
'''


rss_url = 'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany' \
        '&CIK={}&type={}&start={}&count={}&owner=exclude&output=atom' \
        .format(arg_cik, arg_doc_type, start, count)

print("********** Accees the following web page and save it as browse-edgar.txt file onto your working directory where you run this script. **********")
print(rss_url)    #https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001018724&type=10-K&start=0&count=100&owner=exclude&output=atom
time.sleep(t)

#If you omit the following part of the rss_url,
#&output=atom
#then you can see a human-readable page for your reference.
#https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001018724&type=10-K&start=0&count=100&owner=exclude






########## Load a txt file (browse-edgar.txt), delete the first line, and then conver to an xml file

#This txt file contains xml data.
#browse-edgar.txt

'''
If the first two rows of the file browse-edgar.txt is as follows,

<?xml version="1.0" encoding="ISO-8859-1" ?>
  <feed xmlns="http://www.w3.org/2005/Atom">

then try the following:
'''

def read_file(names):
  ans = []
  for name in names:
    with open(name, "r") as f:
        ans.append(f.readlines())
  return ans

def write_file(names, lines):
  for i in range(len(names)):
    with open(names[i], "w") as f:
      for line in lines[i]:
        f.write(line)

lines_ary = read_file(["browse-edgar.txt"])

# Delete the first line of the file browse-edgar.txt
#<?xml version="1.0" encoding="ISO-8859-1" ?>
for lines in lines_ary:
  del(lines[0])    #0 means 1st line of the file to be deleted

write_file(["browse-edgar.xml"], lines_ary)




########## Load an xml file (browse-edgar.xml)

with open("browse-edgar.xml") as f:
    #feed = BeautifulSoup(f, "xml") # xml is a parser
    feed = BeautifulSoup(f, "xml").feed # xml is a parser ('feed' tag is the topmost tag in the xml file)

#print(feed)
print(feed.prettify())


entries = [
        (
            entry.content.find('filing-href').getText(),
            entry.content.find('filing-type').getText(),
            entry.content.find('filing-date').getText())
        for entry in feed.find_all('entry', recursive=False)]



df_CIK_ticker = pd.read_csv('CIK_ticker.csv', dtype=str, header=0, index_col=False)

print(df_CIK_ticker)

#print(df_CIK_ticker[df_CIK_ticker['cik'] == "0001018724"])
#print(df_CIK_ticker[df_CIK_ticker['cik'] == cik])
#
#print(df_CIK_ticker[df_CIK_ticker['cik'] == cik]['ticker'])
#ticker = df_CIK_ticker[df_CIK_ticker['cik'] == cik]['ticker']
#ticker = str(df_CIK_ticker[df_CIK_ticker['cik'] == cik]['ticker'])
ticker = df_CIK_ticker[df_CIK_ticker['cik'] == arg_cik]['ticker'][0]
print(ticker)

sec_data = {}
sec_data[ticker] = entries

#pprint.pprint(sec_data[ticker][:5])
pprint.pprint(sec_data[ticker][:])

'''
[('https://www.sec.gov/Archives/edgar/data/1018724/000101872421000004/0001018724-21-000004-index.htm',
  '10-K',
  '2021-02-03'),
 ('https://www.sec.gov/Archives/edgar/data/1018724/000101872420000004/0001018724-20-000004-index.htm',
  '10-K',
  '2020-01-31'),
 ('https://www.sec.gov/Archives/edgar/data/1018724/000101872419000004/0001018724-19-000004-index.htm',
  '10-K',
  '2019-02-01'),
 ('https://www.sec.gov/Archives/edgar/data/1018724/000101872418000005/0001018724-18-000005-index.htm',
  '10-K',
  '2018-02-02'),
  
  ...
  
 ('https://www.sec.gov/Archives/edgar/data/1018724/0000891020-98-000448-index.html',
  '10-K405',
  '1998-03-30')]
'''
'''
We received a list of urls pointing to files containing metadata related to each filling. Metadata isn’t relevant to us so we pull the filling by replacing the url with the filling url. Let’s view download progress by using tqdm and look at an example document.
'''




########## Download 10-K or other forms of documents on the SEC EDGAR database

raw_fillings_by_ticker     = {}
raw_fillings_by_ticker_url = {}

for ticker, data in sec_data.items():
    #
    raw_fillings_by_ticker[ticker]     = {}
    raw_fillings_by_ticker_url[ticker] = {}
    #
    for index_url, file_type, file_date in tqdm(data, desc='Downloading {} Fillings'.format(ticker), unit='filling'):
        #if (file_type == '10-K'):
        if (file_type == arg_doc_type):
            file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')            
            #
            #raw_fillings_by_ticker[ticker][file_date] = sec_api.get(file_url)
            #raw_fillings_by_ticker[ticker][file_date] = requests.get(file_url, verify=False)
            #
            #print(ticker)
            #print(file_date)
            #print(file_url)
            #print(type(raw_fillings_by_ticker))
            raw_fillings_by_ticker_url[ticker][file_date] = file_url
            #
            #####raw_fillings_by_ticker[ticker][file_date] = requests.get(file_url, verify=False).text

#print(raw_fillings_by_ticker_url)
#print(raw_fillings_by_ticker_url[ticker])
#print(raw_fillings_by_ticker_url[ticker].values())
list_raw_fillings_by_ticker_url = list(raw_fillings_by_ticker_url[ticker].values())
#print(len(list_raw_fillings_by_ticker_url))    #21
#print(list_raw_fillings_by_ticker_url[0])    #https://www.sec.gov/Archives/edgar/data/1018724/000101872421000004/0001018724-21-000004.txt

for i in range(len(list_raw_fillings_by_ticker_url)):
    print(list_raw_fillings_by_ticker_url[i])
with open('list_raw_fillings_by_ticker_url.txt', mode='w') as f:
    f.write('\n'.join(list_raw_fillings_by_ticker_url))

print("")
print("********** Access the URLs above and save as txt files. **********")
print("********** You can open the saved file list_raw_fillings_by_ticker_url.txt and see all the URLs as well. **********")
time.sleep(t)

#print(requests.get('https://www.sec.gov/Archives/edgar/data/1018724/000101872421000004/0001018724-21-000004.txt', verify=False))
#print(requests.get('https://www.sec.gov/Archives/edgar/data/1018724/000101872421000004/0001018724-21-000004.txt', verify=False).text)


list_raw_fillings_by_ticker_url_file = list_raw_fillings_by_ticker_url
#
for i in range(len(list_raw_fillings_by_ticker_url_file)):
    #print(list_raw_fillings_by_ticker_url_file[i])
    list_raw_fillings_by_ticker_url_file[i] = list_raw_fillings_by_ticker_url[i].split('/')[-1]
with open('list_raw_fillings_by_ticker_url_file.txt', mode='w') as f:
    f.write('\n'.join(list_raw_fillings_by_ticker_url_file))
#
#
for i in range(len(list_raw_fillings_by_ticker_url_file)):
    #
    print(list_raw_fillings_by_ticker_url_file[i])
    #
    with open(list_raw_fillings_by_ticker_url_file[i], mode='r') as f:
        dt = f.read()
        #print(dt)
#


for ticker, data in sec_data.items():
    #
    #####raw_fillings_by_ticker[ticker]     = {}
    #####raw_fillings_by_ticker_url[ticker] = {}
    #
    for index_url, file_type, file_date in tqdm(data, desc='Downloading {} Fillings'.format(ticker), unit='filling'):
        #if (file_type == '10-K'):
        if (file_type == arg_doc_type):
            file_url = index_url.replace('-index.htm', '.txt').replace('.txtl', '.txt')            
            #print(file_url)
            print(file_url.split('/')[-1])
            #
            #####raw_fillings_by_ticker[ticker][file_date] = requests.get(file_url, verify=False).text
            with open(file_url.split('/')[-1], mode='r') as f:
                raw_fillings_by_ticker[ticker][file_date] = f.read()


#print('Example Document:\n\n{}...'.format(next(iter(raw_fillings_by_ticker[example_ticker].values()))[:1000]))
print('Example Document:\n\n{}...'.format(next(iter(raw_fillings_by_ticker[ticker].values()))[:1000]))

#print(raw_fillings_by_ticker[ticker])
#print(raw_fillings_by_ticker[ticker].values())




########## Get Documents

'''
With theses fillings downloaded, we want to break them into their associated documents. These documents are sectioned off in the fillings with the tags <DOCUMENT> for the start of each document and </DOCUMENT> for the end of each document. There's no overlap with these documents, so each </DOCUMENT> tag should come after the <DOCUMENT> with no <DOCUMENT> tag in between.

Implement get_documents to return a list of these documents from a filling. Make sure not to include the tag in the returned document text.

'''

import re


def get_documents(text):
    """
    Extract the documents from the text

    Parameters
    ----------
    text : str
        The text with the document strings inside

    Returns
    -------
    extracted_docs : list of str
        The document strings found in `text`
    """
    #
    # TODO: Implement
    extracted_docs = []
    #
    doc_start_pattern = re.compile(r'<DOCUMENT>')
    doc_end_pattern = re.compile(r'</DOCUMENT>')   
    #
    doc_start_is = [x.end() for x in doc_start_pattern.finditer(text)]
    doc_end_is = [x.start() for x in doc_end_pattern.finditer(text)]
    #
    for doc_start_i, doc_end_i in zip(doc_start_is, doc_end_is):
            extracted_docs.append(text[doc_start_i:doc_end_i])
    #
    return extracted_docs


#project_tests.test_get_documents(get_documents)



filling_documents_by_ticker = {}

for ticker, raw_fillings in raw_fillings_by_ticker.items():
    filling_documents_by_ticker[ticker] = {}
    for file_date, filling in tqdm(raw_fillings.items(), desc='Getting Documents from {} Fillings'.format(ticker), unit='filling'):
        filling_documents_by_ticker[ticker][file_date] = get_documents(filling)


print('\n\n'.join([
    'Document {} Filed on {}:\n{}...'.format(doc_i, file_date, doc[:200])
    #for file_date, docs in filling_documents_by_ticker[example_ticker].items()
    for file_date, docs in filling_documents_by_ticker[ticker].items()
    for doc_i, doc in enumerate(docs)][:3]))




########## Get Document Types
'''
Now that we have all the documents, we want to find the 10-k form in this 10-k filing. Implement the get_document_type function to return the type of document given. The document type is located on a line with the <TYPE> tag. 
'''

def get_document_type(doc):
    """
    Return the document type lowercased

    Parameters
    ----------
    doc : str
        The document string

    Returns
    -------
    doc_type : str
        The document type lowercased
    """
    #
    # TODO: Implement
    type_pattern = re.compile(r'<TYPE>[^\n]+')
    #
    doc_type = type_pattern.findall(doc)[0][len('<TYPE>'):] 
    #
    return doc_type.lower()


#project_tests.test_get_document_type(get_document_type)


'''
print(df_CIK_ticker)
cik = df_CIK_ticker[df_CIK_ticker['cik'] == arg_cik]['cik'][0]
print(cik)
'''

ten_ks_by_ticker = {}

for ticker, filling_documents in filling_documents_by_ticker.items():
    ten_ks_by_ticker[ticker] = []
    for file_date, documents in filling_documents.items():
        for document in documents:
            #if get_document_type(document) == '10-k':
            if get_document_type(document) == arg_doc_type.lower():
                ten_ks_by_ticker[ticker].append({
                    #'cik': cik_lookup[ticker],
                    'cik': arg_cik,
                    'file': document,
                    'file_date': file_date})

#project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['cik', 'file', 'file_date'])
project_helper.print_ten_k_data(ten_ks_by_ticker[ticker][:5], ['cik', 'file', 'file_date'])




########## Preprocess the Data

##### Clean Up

def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    #
    return text


def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    #
    return text


for ticker, ten_ks in ten_ks_by_ticker.items():
    #for ten_k in tqdm(ten_ks, desc='Cleaning {} 10-Ks'.format(ticker), unit='10-K'):
    for ten_k in tqdm(ten_ks, desc='Cleaning {} '.format(ticker), unit=arg_doc_type):
        ten_k['file_clean'] = clean_text(ten_k['file'])


#project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['file_clean'])
project_helper.print_ten_k_data(ten_ks_by_ticker[ticker][:5], ['file_clean'])



##### Lemmatize

def lemmatize_words(words):
    """
    Lemmatize words 

    Parameters
    ----------
    words : list of str
        List of words

    Returns
    -------
    lemmatized_words : list of str
        List of lemmatized words
    """
    #
    # TODO: Implement
    lemmatized_words = [WordNetLemmatizer().lemmatize(word, 'v') for word in words]
    #
    return lemmatized_words


#project_tests.test_lemmatize_words(lemmatize_words)



word_pattern = re.compile('\w+')

for ticker, ten_ks in ten_ks_by_ticker.items():
    for ten_k in tqdm(ten_ks, desc='Lemmatize {} 10-Ks'.format(ticker), unit='10-K'):
        ten_k['file_lemma'] = lemmatize_words(word_pattern.findall(ten_k['file_clean']))


#project_helper.print_ten_k_data(ten_ks_by_ticker[example_ticker][:5], ['file_lemma'])
project_helper.print_ten_k_data(ten_ks_by_ticker[ticker][:5], ['file_lemma'])



##### Remove Stopwords




lemma_english_stopwords = lemmatize_words(stopwords.words('english'))

for ticker, ten_ks in ten_ks_by_ticker.items():
    #for ten_k in tqdm(ten_ks, desc='Remove Stop Words for {} 10-Ks'.format(ticker), unit='10-K'):
    for ten_k in tqdm(ten_ks, desc='Remove Stop Words for {} '.format(ticker), unit=arg_doc_type):
        ten_k['file_lemma'] = [word for word in ten_k['file_lemma'] if word not in lemma_english_stopwords]


print('Stop Words Removed')




########## Analysis on 10ks

##### Loughran McDonald Sentiment Word Lists
'''
We'll be using the Loughran and McDonald sentiment word lists. These word lists cover the following sentiment:

Negative
Positive
Uncertainty
Litigious
Constraining
Superfluous
Modal

This will allow us to do the sentiment analysis on the 10-ks. Let's first load these word lists. We'll be looking into a few of these sentiments.
'''


sentiments = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining', 'interesting']


#sentiment_df = pd.read_csv(os.path.join('..', '..', 'data', 'project_5_loughran_mcdonald', 'loughran_mcdonald_master_dic_2016.csv'))
#
sentiment_df = pd.read_csv('loughran_mcdonald_master_dic_2016.csv')
#
#You can download this file (loughran_mcdonald_master_dic_2016.csv) and save it onto the same directory with this script.
#https://github.com/soheil-mpg/NLP-on-Financial-Statements/blob/master/loughran_mcdonald_master_dic_2016.csv


sentiment_df.columns = [column.lower() for column in sentiment_df.columns] # Lowercase the columns for ease of use

# Remove unused information
sentiment_df = sentiment_df[sentiments + ['word']]
sentiment_df[sentiments] = sentiment_df[sentiments].astype(bool)
sentiment_df = sentiment_df[(sentiment_df[sentiments]).any(1)]

# Apply the same preprocessing to these words as the 10-k words
sentiment_df['word'] = lemmatize_words(sentiment_df['word'].str.lower())
sentiment_df = sentiment_df.drop_duplicates('word')


sentiment_df.head()


##### Bag of Words
'''
using the sentiment word lists, let's generate sentiment bag of words from the 10-k documents. Implement get_bag_of_words to generate a bag of words that counts the number of sentiment words in each doc. You can ignore words that are not in sentiment_words.
'''




from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer


def get_bag_of_words(sentiment_words, docs):
    """
    Generate a bag of words from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """
    #
    # TODO: Implement
    vec = CountVectorizer(vocabulary=sentiment_words)
    vectors = vec.fit_transform(docs)
    words_list = vec.get_feature_names()
    bag_of_words = np.zeros([len(docs), len(words_list)])
    #
    for i in range(len(docs)):
        bag_of_words[i] = vectors[i].toarray()[0]
    #
    return bag_of_words.astype(int)


project_tests.test_get_bag_of_words(get_bag_of_words)




sentiment_bow_ten_ks = {}

for ticker, ten_ks in ten_ks_by_ticker.items():
    lemma_docs = [' '.join(ten_k['file_lemma']) for ten_k in ten_ks]
    #
    sentiment_bow_ten_ks[ticker] = {
        sentiment: get_bag_of_words(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs)
        for sentiment in sentiments}


#Output:
#sentiment_bow_ten_ks.txt
#
#project_helper.print_ten_k_data([sentiment_bow_ten_ks[example_ticker]], sentiments)
project_helper.print_ten_k_data([sentiment_bow_ten_ks[ticker]], sentiments)
#
print(sentiment_bow_ten_ks[ticker])
print(type(sentiment_bow_ten_ks[ticker]))    #<class 'dict'>


##### Jaccard Similarity
'''
Using the bag of words, let's calculate the jaccard similarity on the bag of words and plot it over time. Implement get_jaccard_similarity to return the jaccard similarities between each tick in time. Since the input, bag_of_words_matrix, is a bag of words for each time period in order, you just need to compute the jaccard similarities for each neighboring bag of words. Make sure to turn the bag of words into a boolean array when calculating the jaccard similarity.
'''





def get_jaccard_similarity(bag_of_words_matrix):
    """
    Get jaccard similarities for neighboring documents

    Parameters
    ----------
    bag_of_words : 2-d Numpy Ndarray of int
        Bag of words sentiment for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    jaccard_similarities : list of float
        Jaccard similarities for neighboring documents
    """
    #
    # TODO: Implement
    jaccard_similarities = []
    bag_of_words_matrix = np.array(bag_of_words_matrix, dtype=bool)
    #
    for i in range(len(bag_of_words_matrix)-1):
            u = bag_of_words_matrix[i]
            v = bag_of_words_matrix[i+1]
            #jaccard_similarities.append(jaccard_similarity_score(u,v))
            #jaccard_similarities.append(jaccard_score(u,v,pos_label = False))
            jaccard_similarities.append(jaccard_score(u,v))
    #
    return jaccard_similarities


#project_tests.test_get_jaccard_similarity(get_jaccard_similarity)



# Get dates for the universe
file_dates = {
    ticker: [ten_k['file_date'] for ten_k in ten_ks]
    for ticker, ten_ks in ten_ks_by_ticker.items()}  

jaccard_similarities = {
    ticker: {
        sentiment_name: get_jaccard_similarity(sentiment_values)
        for sentiment_name, sentiment_values in ten_k_sentiments.items()}
    for ticker, ten_k_sentiments in sentiment_bow_ten_ks.items()}


project_helper.plot_similarities(
    #[jaccard_similarities[example_ticker][sentiment] for sentiment in sentiments],
    [jaccard_similarities[ticker][sentiment] for sentiment in sentiments],
    #file_dates[example_ticker][1:],
    file_dates[ticker][1:],
    #'Jaccard Similarities for {} Sentiment'.format(example_ticker),
    'Jaccard Similarities for {} Sentiment'.format(ticker),
    sentiments)


##### TFIDF
'''
using the sentiment word lists, let's generate sentiment TFIDF from the 10-k documents. Implement get_tfidf to generate TFIDF from each document, using sentiment words as the terms. You can ignore words that are not in sentiment_words.
'''


def get_tfidf(sentiment_words, docs):
    """
    Generate TFIDF values from documents for a certain sentiment

    Parameters
    ----------
    sentiment_words: Pandas Series
        Words that signify a certain sentiment
    docs : list of str
        List of documents used to generate bag of words

    Returns
    -------
    tfidf : 2-d Numpy Ndarray of float
        TFIDF sentiment for each document
        The first dimension is the document.
        The second dimension is the word.
    """
    #
    # TODO: Implement
    vec = TfidfVectorizer(vocabulary=sentiment_words)
    tfidf = vec.fit_transform(docs)
    #
    return tfidf.toarray()


project_tests.test_get_tfidf(get_tfidf)



sentiment_tfidf_ten_ks = {}

for ticker, ten_ks in ten_ks_by_ticker.items():
    lemma_docs = [' '.join(ten_k['file_lemma']) for ten_k in ten_ks]
    #
    sentiment_tfidf_ten_ks[ticker] = {
        sentiment: get_tfidf(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs)
        for sentiment in sentiments}

    
#project_helper.print_ten_k_data([sentiment_tfidf_ten_ks[example_ticker]], sentiments)
project_helper.print_ten_k_data([sentiment_tfidf_ten_ks[ticker]], sentiments)




##### Cosine Similarity
'''
Using the TFIDF values, we'll calculate the cosine similarity and plot it over time. Implement get_cosine_similarity to return the cosine similarities between each tick in time. Since the input, tfidf_matrix, is a TFIDF vector for each time period in order, you just need to computer the cosine similarities for each neighboring vector.
'''

def get_cosine_similarity(tfidf_matrix):
    """
    Get cosine similarities for each neighboring TFIDF vector/document

    Parameters
    ----------
    tfidf : 2-d Numpy Ndarray of float
        TFIDF sentiment for each document
        The first dimension is the document.
        The second dimension is the word.

    Returns
    -------
    cosine_similarities : list of float
        Cosine similarities for neighboring documents
    """
    #
    # TODO: Implement
    cosine_similarities = []    
    #
    for i in range(len(tfidf_matrix)-1):
        cosine_similarities.append(cosine_similarity(tfidf_matrix[i].reshape(1, -1),tfidf_matrix[i+1].reshape(1, -1))[0,0])
    #
    return cosine_similarities


project_tests.test_get_cosine_similarity(get_cosine_similarity)



cosine_similarities = {
    ticker: {
        sentiment_name: get_cosine_similarity(sentiment_values)
        for sentiment_name, sentiment_values in ten_k_sentiments.items()}
    for ticker, ten_k_sentiments in sentiment_tfidf_ten_ks.items()}


project_helper.plot_similarities(
    #[cosine_similarities[example_ticker][sentiment] for sentiment in sentiments],
    [cosine_similarities[ticker][sentiment] for sentiment in sentiments],
    #file_dates[example_ticker][1:],
    file_dates[ticker][1:],
    #'Cosine Similarities for {} Sentiment'.format(example_ticker),
    'Cosine Similarities for {} Sentiment'.format(ticker),
    sentiments)




########## Evaluate Alpha Factors

#pricing = pd.read_csv('../../data/project_5_yr/yr-quotemedia.csv', parse_dates=['date'])
pricing = pd.read_csv('yr-quotemedia.csv', parse_dates=['date'])
pricing = pricing.pivot(index='date', columns='ticker', values='adj_close')

#pricing
print(pricing)


##### Dict to DataFrame

cosine_similarities_df_dict = {'date': [], 'ticker': [], 'sentiment': [], 'value': []}


for ticker, ten_k_sentiments in cosine_similarities.items():
    for sentiment_name, sentiment_values in ten_k_sentiments.items():
        for sentiment_values, sentiment_value in enumerate(sentiment_values):
            cosine_similarities_df_dict['ticker'].append(ticker)
            cosine_similarities_df_dict['sentiment'].append(sentiment_name)
            cosine_similarities_df_dict['value'].append(sentiment_value)
            cosine_similarities_df_dict['date'].append(file_dates[ticker][1:][sentiment_values])

cosine_similarities_df = pd.DataFrame(cosine_similarities_df_dict)
cosine_similarities_df['date'] = pd.DatetimeIndex(cosine_similarities_df['date']).year
cosine_similarities_df['date'] = pd.to_datetime(cosine_similarities_df['date'], format='%Y')


#cosine_similarities_df.head()
print(cosine_similarities_df.head())


##### Alphalens Format

factor_data = {}
skipped_sentiments = []

for sentiment in sentiments:
    cs_df = cosine_similarities_df[(cosine_similarities_df['sentiment'] == sentiment)]
    cs_df = cs_df.pivot(index='date', columns='ticker', values='value')
    #
    try:
        data = al.utils.get_clean_factor_and_forward_returns(cs_df.stack(), pricing.loc[cs_df.index], quantiles=5, bins=None, periods=[1])
        factor_data[sentiment] = data
    except:
        skipped_sentiments.append(sentiment)
    #
if skipped_sentiments:
    print('\nSkipped the following sentiments:\n{}'.format('\n'.join(skipped_sentiments)))
    
#factor_data[sentiments[0]].head()


##### Alphalens Format with Unix Time

'''
Alphalen's factor_rank_autocorrelation and mean_return_by_quantile functions require unix timestamps to work, so we'll also create factor dataframes with unix time.
'''
unixt_factor_data = {
    factor: data.set_index(pd.MultiIndex.from_tuples(
        [(x.timestamp(), y) for x, y in data.index.values],
        names=['date', 'asset']))
    for factor, data in factor_data.items()}


##### Factor Returns
'''
Let's view the factor returns over time. We should be seeing it generally move up and to the right.
'''

ls_factor_returns = pd.DataFrame()

for factor_name, data in factor_data.items():
    ls_factor_returns[factor_name] = al.performance.factor_returns(data).iloc[:, 0]

### To avoid an error here, there needs to be multiple tickers as the reference does.
#(1 + ls_factor_returns).cumprod().plot()


##### Basis Points Per Day per Quantile
'''
It is not enough to look just at the factor weighted return. A good alpha is also monotonic in quantiles. Let's looks the basis points for the factor returns.
'''

qr_factor_returns = pd.DataFrame()

for factor_name, data in unixt_factor_data.items():
    qr_factor_returns[factor_name] = al.performance.mean_return_by_quantile(data)[0].iloc[:, 0]

### To avoid an error here, there needs to be multiple tickers as the reference does.
'''
(10000*qr_factor_returns).plot.bar(
    subplots=True,
    sharey=True,
    layout=(5,3),
    figsize=(14, 14),
    legend=False)
'''

##### Turnover Analysis
'''
Without doing a full and formal backtest, we can analyze how stable the alphas are over time. Stability in this sense means that from period to period, the alpha ranks do not change much. Since trading is costly, we always prefer, all other things being equal, that the ranks do not change significantly per period. We can measure this with the Factor Rank Autocorrelation (FRA).
'''

ls_FRA = pd.DataFrame()

for factor, data in unixt_factor_data.items():
    ls_FRA[factor] = al.performance.factor_rank_autocorrelation(data)

### To avoid an error here, there needs to be multiple tickers as the reference does.
'''
ls_FRA.plot(title="Factor Rank Autocorrelation")
'''


##### Sharpe Ratio of the Alphas
'''
The last analysis we'll do on the factors will be sharpe ratio. Let's see what the sharpe ratio for the factors are. Generally, a Sharpe Ratio of near 1.0 or higher is an acceptable single alpha for this universe.
'''

daily_annualization_factor = np.sqrt(252)

(daily_annualization_factor * ls_factor_returns.mean() / ls_factor_returns.std()).round(2)