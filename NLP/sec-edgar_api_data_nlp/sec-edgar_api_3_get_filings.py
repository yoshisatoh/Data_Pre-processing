#################### Data Pre-processing: Natural language processing (NLP) for SEC EDGAR database API - 2. Preprocess JSON meta data for each entity with a unique CIK ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/07
# Last Updated: 2021/10/07
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sec-edgar_api_data_nlp/sec-edgar_api_3_get_filings.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/sec-edgar_api_data_nlp/sec-edgar_api_3_get_filings.py
#
#
########## Input Data File(s)
#
#CIKs.txt    # a list of company entitys’ 10-digit Central Index Key (CIK)
#
#csv files as in:
#str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.csv'
#For instance,
#0001018724/CIK0001018724.csv
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#python sec-edgar_api_3_get_filings.py CIKs.txt
#
#Generally,
#python sec-edgar_api_3_get_filings.py (arg_CIKs_file_name)
#
#
########## Output Data File(s)
#
#txt files as in:
#str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.txt'
#For instance,
#0001018724/CIK0001018724.txt
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

import requests
'''
import nltk
import numpy as np
import pandas as pd
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




########## Load all the csv files for each entity with a unique CIK. Each csv file contains all the URLs for the documents in the specified form (e.g., 10-K).

# CIKs: A list of CIKs for all the entities.

for l in range(len(CIKs)):
    #
    #Load a csv file
    df = pd.read_csv(str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.csv', dtype=object, header=0, index_col=None)
    #
    #print(df)
    #print(df['URL'])
    #
    df['URL'].to_csv(str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.txt', sep=' ', header=None, index=False)
    #
    for t in range(len(df['URL'])):
        print(df['URL'][t])
        #
        #Get a text file for an entity one by one
        #txtf = requests.get(df['URL'][t], allow_redirects=True)    #If you see an error of [SSL: CERTIFICATE_VERIFY_FAILED], try the following:
        txtf = requests.get(df['URL'][t], allow_redirects=True, verify=False)
        #
        #Save text file onto the directory with the name of the CIK
        #print(str(CIKs[l]) + '/' + str(df['URL'][t].split('/')[-1]) + '.txt')
        open(str(CIKs[l]) + '/' + str(df['URL'][t].split('/')[-1]), 'wb').write(txtf.content)
        #
        #
        time.sleep(60/10+1)
        #Note that the SEC will limit automated searches to a total of no more than 10 requests per second.
        #https://www.sec.gov/oit/announcement/new-rate-control-limits
    #


print('')
print('If you find that downloaded txt files are invalid, then access txt files as in:')
for l in range(len(CIKs)):
    print(str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.txt')
print('')
print('Open each txt file for each entity with a unique CIK, and access all the URLs written in each txt file.')
print('Then save all the URLs on the sec.gov as txt files.')
print('')
