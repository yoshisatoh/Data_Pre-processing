#################### Data Pre-processing: Natural language processing (NLP) for SEC EDGAR database API - 4. Preprocess filings txt data for each entity with a unique CIK ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/08
# Last Updated: 2021/10/08
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sec-edgar_api_data_nlp/sec-edgar_api_4_preprocess_filings.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/sec-edgar_api_data_nlp/sec-edgar_api_4_preprocess_filings.py
#
#
########## Input Data File(s)
#
#CIKs.txt    # a list of company entitys’ 10-digit Central Index Key (CIK)
#
#csv files with URLs of documents:
#str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.csv'
#For instance,
#0001018724/CIK0001018724.csv
#
#loughran_mcdonald_master_dic_2016.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#python sec-edgar_api_4_preprocess_filings.py CIKs.txt 10-K
#
#Generally,
#python sec-edgar_api_4_preprocess_filings.py (arg_CIKs_file_name) (arg_doc_type)
#
#
########## Output Data File(s)
#
#[1]  A list of Individual Filing Files
#txt files as in:
#str(CIKs[l]) + '/' + str(lst[i])
#For instance,
#0001018724/0001018724.txt

#[2] Individual Filing Files
#txt files as in:
#str(CIKs[l]) + '/' + str(lst[i]) + '_' + str(arg_doc_type) + '_' + str(d) + '.txt'
#For instance,
#0001018724/0001018724-21-000004.txt_10-K_0.txt
#
#[3] A list of sub-documents with a specified form (arg_doc_type, e.g., 10-K)
#str(CIKs[l]) + '/' + str(arg_doc_type)
#For instance,
#0001018724/10-K.txt
#
#[4] Combined Dictionary of Word Frequency and Loughran McDonald Sentiment Word Lists for each sub-document with a specified form (arg_doc_type, e.g., 10-K)
#str(CIKs[l]) + '/' + str(lst[i]) + '.csv'
#For instance,
#0001018724-21-000004.txt_10-K_0.txt.csv
#
#[5] sentiment_df.csv
#
#
########## References
#
#EDGAR | Company Filings
#https://www.sec.gov/edgar/searchedgar/companysearch.html
#
#EDGAR Application Programming Interfaces
#https://www.sec.gov/edgar/sec-api-documentation
#
#NLP in the Stock Market
#https://towardsdatascience.com/nlp-in-the-stock-market-8760d062eb92
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
import time

import pandas as pd

#import requests
import re

from bs4 import BeautifulSoup

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np

import collections
'''
import matplotlib.pyplot as plt
'''

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import warnings
warnings.simplefilter('ignore')




########## argument(s)

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #sec-edgar_api_nlp.py

#SEC EDGAR database
arg_CIKs_file_name = str(sys.argv[1])    #CIKs.txt
arg_doc_type       = str(sys.argv[2])    #10-K




########## SEC EDGAR database
#
#SEC EDGAR database provides various forms of filings.
#
#10-K forms are annual reports filed by companies to provide a comprehensive summary of their financial performance (these reports are mandated by the Securities and Exchange Commission). 
#Combing through these reports is often tedious for investors.
#Through sentiment analysis, a subfield of natural language processing, investors can quickly understand if the tone of the report is positive, negative, or litigious etc.
#The overall sentiment expressed in the 10-k form can then be used to help investors decide if they should invest in the company.
#
#10-Q forms are QUARTERLY reports.


CIKs = []    # an empty list of CIKs
#
with open(arg_CIKs_file_name, mode='r') as f:
    for l in f.readlines():
        l = l.rstrip()
        #print(l)
        CIKs.append(l)
#print(f.closed)    #True




########## Output filings txt file names as str(CIKs[l]) + '/' + str(CIKs[l]) + '.txt'

# CIKs: A list of CIKs for all the entities.

for l in range(len(CIKs)):
    #
    ##### Load all the csv files for each entity with a unique CIK. Each csv file contains all the URLs for the documents in the specified form (e.g., 10-K).
    with open(str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.txt', mode='r') as f:
        lst = f.read().splitlines()
        #print(lst)
        #
        for i in range(len(lst)):
            #print(i)
            #print(lst[i].split('/')[-1])
            lst[i] = lst[i].split('/')[-1]
    #print(tf.closed)
    #
    ###### Ouput filings txt file names
    with open(str(CIKs[l]) + '/' + str(CIKs[l]) + '.txt', mode='w') as f:
        for i in lst:
            f.write("%s\n" % i)





########## Get Documents (or put it differently, extract documents of a specified-form from each txt file per entity per year.)
'''
With theses fillings downloaded, we want to break them into their associated documents. These documents are sectioned off in the fillings with the tags <DOCUMENT> for the start of each document and </DOCUMENT> for the end of each document. There's no overlap with these documents, so each </DOCUMENT> tag should come after the <DOCUMENT> with no <DOCUMENT> tag in between.

Implement get_documents to return a list of these documents from a filling. Make sure not to include the tag in the returned document text.
'''


# Each text file has multiple <DOCUMENT> ... </DOCUMENT> sections. The function get_documents() decomposes each text file into DOCUMENT sections
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


# Each text file after removing <DOCUMENT> ... </DOCUMENT> section starts from a <TYPE> tag, such as '<TYPE>10-K'.  This function is to specify text files with a certain <TYPE>.
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
    #return doc_type.lower()
    return doc_type.upper()


##### extract documents of a specified-form from each txt file per entity per year.
for l in range(len(CIKs)):
    if os.path.exists(str(CIKs[l]) + '/' + str(arg_doc_type) + '.txt'):
        os.remove(str(CIKs[l]) + '/' + str(arg_doc_type) + '.txt')    #if a text file with the same name exists, then this part deletes the file first


#print('Loading following txt files.')
#print('CIK' + ' : ' + 'txt file')
#print('--------------------------------------------------')
for l in range(len(CIKs)):
    #
    ##### Load all the csv file names for each entity with a unique CIK.
    with open(str(CIKs[l]) + '/' + str(CIKs[l]) + '.txt', mode='r') as f:
        lst = f.read().splitlines()
        #
        for i in range(len(lst)):
            #print(lst[i])
            print(str(CIKs[l]) + '/' + str(lst[i]))    # each text file
            #
            with open(str(CIKs[l]) + '/' + str(lst[i]), mode='r') as tf:    # open each text file, which has multiple <DOCUMENT> ... </DOCUMENT> sections
                #print(tf.read())
                #print(get_documents(tf.read()))
                docs = get_documents(tf.read())
                #print(len(docs))    #number of <DOCUMENT>
                #
                for d in range(len(docs)):    # d is number of documents in each text file (e.g., 0001018724-14-000006.txt); if arg_doc_type = '10-K', then it'll be 1 as 10-K forms are ANNUAL report
                    #print(docs[d])    #print each section of <DOCUMENT> ... </DOCUMENT>
                    #
                    #print(get_document_type(docs[d]))    #get document types in each text files
                    #
                    if get_document_type(docs[d]) == arg_doc_type:
                        print(arg_doc_type)
                        #print(type(docs[d]))
                        #print(docs[d])    # a document with a specified form (arg_doc_type, e.g., 10-K)
                        #
                        with open(str(CIKs[l]) + '/' + str(arg_doc_type) + '.txt', 'a') as arg_doc_type_list_f:
                            #
                            with open(str(CIKs[l]) + '/' + str(lst[i]) + '_' + str(arg_doc_type) + '_' + str(d) + '.txt', 'w') as arg_doc_type_f:
                                arg_doc_type_f.write(docs[d])
                            #arg_doc_type_list_f.write(str(lst[i]) + '_' + str(arg_doc_type) + '_' + str(d) + '.txt\n')
                            #print(str(lst[i]) + '_' + str(arg_doc_type) + '_' + str(d) + '.txt\n')
                            print(str(lst[i]) + '_' + str(arg_doc_type) + '_' + str(d) + '.txt', file=arg_doc_type_list_f)



########## Preprocess the Data (individual documents of a specified form, e.g., 10-K)

#####Clean Up
#As you can see, the text for the documents are very messy. To clean this up, we'll remove the html and lowercase all the text.


def remove_html_tags(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    #
    return text


def clean_text(text):
    text = text.lower()
    text = remove_html_tags(text)
    #
    return text


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






def print_ten_k_data(ten_k_data, fields, field_length_limit=50):
    indentation = '  '
    #
    print('[')
    for ten_k in ten_k_data:
        print_statement = '{}{{'.format(indentation)
        for field in fields:
            value = str(ten_k[field])
            #
            # Show return lines in output
            if isinstance(value, str):
                value_str = '\'{}\''.format(value.replace('\n', '\\n'))
            else:
                value_str = str(value)
            #
            # Cut off the string if it gets too long
            if len(value_str) > field_length_limit:
                value_str = value_str[:field_length_limit] + '...'
            #
            print_statement += '\n{}{}: {}'.format(indentation * 2, field, value_str)
        #
        print_statement += '},'
        print(print_statement)
    print(']')






# Load a per-entity list of all the txt file names in a specific form, e.g., 10-K
for l in range(len(CIKs)):
    #    
    with open(str(CIKs[l]) + '/' + str(arg_doc_type) + '.txt', mode='r') as f:
        lst = f.read().splitlines()
        #print(lst)
        #
        print(str(CIKs[l]) + ':' + str(arg_doc_type))
        for i in range(len(lst)):
            #print(lst[i])
            #print('', end='')
            print(str(CIKs[l]) + '/' + str(lst[i]))    # each text file is an individual document in a specific form like 10-K
            #
            #
            #Call clean_text to parse html tags and make all the text lower cases.
            with open (str(CIKs[l]) + '/' + str(lst[i]), mode='r') as ff:
                cnt = ff.read()
                #print(clean_text(cnt))
                cnt_cleaned = clean_text(cnt)
                #print(cnt_cleaned)
            #
            ##### Lemmatize
            '''
            With the text cleaned up, it's time to distill the verbs down. Implement the lemmatize_words function to lemmatize verbs in the list of words provided.
            '''
            word_pattern = re.compile('\w+')
            #print(lemmatize_words(word_pattern.findall(cnt_cleaned)))
            cnt_cleaned_lemmatized = lemmatize_words(word_pattern.findall(cnt_cleaned))
            #
            #
            ##### Remove Stopwords
            lemma_english_stopwords = lemmatize_words(stopwords.words('english'))
            #
            cnt_cleaned_lemmatized = [word for word in cnt_cleaned_lemmatized if word not in lemma_english_stopwords]
            #print(cnt_cleaned_lemmatized)
            #print(type(cnt_cleaned_lemmatized))    #<class 'list'>
            #
            #
            ##### Create a Dictionary of Word Frequency
            #
            lst_word_freq = {}
            c = collections.Counter(cnt_cleaned_lemmatized)
            #print(c.most_common())
            lst_word_freq = c.most_common()
            #print(type(lst_word_freq))    #<class 'list'>
            #print(len(lst_word_freq))
            #print(lst_word_freq[0])
            #print(lst_word_freq[0][0])
            df_word_freq = pd.DataFrame(lst_word_freq, columns=["word", "count"], index=None)
            print(df_word_freq)
            #
            #
            ##### Loughran McDonald Sentiment Word Lists
            #We'll be using the Loughran and McDonald sentiment word lists. These word lists cover the following sentiment:
            #
            #Negative
            #Positive
            #Uncertainty
            #Litigious
            #Constraining
            #Superfluous0
            #Modal
            #
            #
            #This will allow us to do the sentiment analysis on the documents of the specified form like 10-Ks. Let's first load these word lists. We'll be looking into a few of these sentiments.
            sentiments = ['negative', 'positive', 'uncertainty', 'litigious', 'constraining', 'interesting']
            #
            sentiment_df = pd.read_csv('loughran_mcdonald_master_dic_2016.csv')
            sentiment_df.columns = [column.lower() for column in sentiment_df.columns] # Lowercase the columns for ease of use
            #
            # Remove unused information
            sentiment_df = sentiment_df[sentiments + ['word']]
            sentiment_df[sentiments] = sentiment_df[sentiments].astype(bool)
            sentiment_df = sentiment_df[(sentiment_df[sentiments]).any(1)]
            #
            # Apply the same preprocessing to these words as the 10-k words
            sentiment_df['word'] = lemmatize_words(sentiment_df['word'].str.lower())
            sentiment_df = sentiment_df.drop_duplicates('word')
            #
            print(sentiment_df.head())
            '''
                negative  positive  uncertainty  litigious  constraining  interesting          word
            9       True     False        False      False         False        False       abandon
            12      True     False        False      False         False        False   abandonment
            13      True     False        False      False         False        False  abandonments
            51      True     False        False      False         False        False      abdicate
            54      True     False        False      False         False        False    abdication
            '''
            sentiment_df.to_csv('sentiment_df.csv', header=True, index=False)
            #
            #
            #
            #
            ##### Bag of Words
            #using the sentiment word lists, let's generate sentiment bag of words from the documents of the specified form such as 10-K.
            #Implement get_bag_of_words to generate a bag of words that counts the number of sentiment words in each doc. You can ignore words that are not in sentiment_words.
            #
            '''
            cnt_cleaned_lemmatized_bow = {
                #sentiment: get_bag_of_words(sentiment_df[sentiment_df[sentiment]]['word'], lemma_docs) for sentiment in sentiments
                sentiment: get_bag_of_words(sentiment_df[sentiment_df[sentiment]]['word'], cnt_cleaned_lemmatized) for sentiment in sentiments
            }
            print_ten_k_data([cnt_cleaned_lemmatized_bow], sentiments)
            '''
            #
            #
            #
            #
            ##### Combine a Dictionary of Word Frequency with Loughran McDonald Sentiment Word Lists
            #
            print(pd.merge(df_word_freq, sentiment_df, on='word', how='inner'))
            print(str(CIKs[l]) + '/' + str(lst[i]) + '.csv')
            #
            df_word_freq_sentiment = pd.merge(df_word_freq, sentiment_df, on='word', how='inner')
            #
            with open (str(CIKs[l]) + '/' + str(lst[i]) + '.csv', mode='wt', newline="") as fff:
                df_word_freq_sentiment.to_csv(fff, index=False, header=True)
            
            
            
            
            
            

            


