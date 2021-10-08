#################### Data Pre-processing: Natural language processing (NLP) for SEC EDGAR database API - 2. Preprocess JSON meta data for each entity with a unique CIK, and then get URLs for documents in the specified form ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/07
# Last Updated: 2021/10/08
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sec-edgar_api_data_nlp/sec-edgar_api_2_get_filingsURLs.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/sec-edgar_api_data_nlp/sec-edgar_api_2_get_filingsURLs.py
#
#
########## Input Data File(s)
#
#CIKs.txt    # a list of company entitys’ 10-digit Central Index Key (CIK)
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#python sec-edgar_api_2_get_filingsURLs.py CIKs.txt 10-K
#
#Generally,
#python sec-edgar_api_2_get_filingsURLs.py (arg_CIKs_file_name) (arg_doc_type)
#
#
########## Output Data File(s)
#
#csv files as in:
#str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.csv'
#
#For instance,
#0001018724/CIK0001018724.csv
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

import json

import pandas as pd

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




########## Load all the entitys' json meta data of filings.

# CIKs: A list of CIKs for all the entities.

for l in range(len(CIKs)):
    #
    #Load a json metadata
    with open(str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.json', 'r') as f:    #Confirm that each json file has prefix "CIK"
        jsn = json.load(f)
        #
        #print(type(jsn))    #<class 'dict'>
        #
        #print(json.dumps(jsn, indent=4, sort_keys=True))
        #
        #show all elements of json
        for jsn_key in jsn:
            #print(jsn_key)
            print('', end='')
            '''
            as of 7th October 2021, a list of all the json keys are as follows:
            
            cik
            entityType
            sic
            sicDescription
            insiderTransactionForOwnerExists
            insiderTransactionForIssuerExists
            name
            tickers
            exchanges
            ein
            description
            website
            investorWebsite
            category
            fiscalYearEnd
            stateOfIncorporation
            stateOfIncorporationDescription
            addresses
            phone
            flags
            formerNames
            filings
            '''
            #
        #
        #print(jsn['cik'])
        #
        #print(jsn['tickers'])
        #
        #print(jsn['filings'])
        for jsn_key in jsn['filings']:
            #print(jsn_key)
            '''
            recent
            files
            '''
            print('', end='')
        #
        for jsn_key in jsn['filings']['recent']:
            #print(jsn_key)
            '''
            accessionNumber
            filingDate
            reportDate
            acceptanceDateTime
            act
            form
            fileNumber
            filmNumber
            items
            size
            isXBRL
            isInlineXBRL
            primaryDocument
            primaryDocDescription            
            '''
            print('', end='')
        #
        for jsn_key in jsn['filings']['recent']['form']:
            #print(jsn_key)
            #if jsn_key == arg_doc_type:
            #    print(jsn_key)    # show records with the specified form (arg_doc_type, e.g., 10-K)
            '''
            '''
            print('', end='')
        #
        #
        '''
        print(jsn['cik'])
        print(jsn['tickers'])
        #print(jsn['filings']['files'][0]['name'])
        #print(jsn['filings']['recent']['accessionNumber'])
        #print(jsn['filings']['recent']['filingDate'])
        #print(jsn['filings']['recent']['reportDate'])
        #print(jsn['filings']['recent']['acceptanceDateTime'])
        print(jsn['filings']['recent']['form'])
        print(jsn['filings']['recent']['fileNumber'])
        print(jsn['filings']['recent']['filmNumber'])
        ##print(jsn['filings']['recent']['primaryDocument'])
        ##print(jsn['filings']['recent']['primaryDocDescription'])
        '''
        #
        '''
        print(jsn['cik'])
        print(type(jsn['cik']))    #<class 'str'>
        #
        print(jsn['tickers'])
        print(len(jsn['tickers']))
        print(type(jsn['tickers']))    #<class 'list'>
        #
        print(jsn['tickers'][0])
        print(type(jsn['tickers'][0]))    #<class 'str'>
        #
        print(len(jsn['filings']['recent']['form']))
        print(type(jsn['filings']['recent']['form']))    #<class 'list'>
        '''
        #
        df                    = pd.DataFrame(jsn['filings']['recent']['form'], index=None, columns = ['form'])
        df['cik']             = str(jsn['cik']).zfill(10)
        df['tickers']         = jsn['tickers'][0]
        df['accessionNumber'] = jsn['filings']['recent']['accessionNumber']
        df['filingDate']      = jsn['filings']['recent']['filingDate']
        #df['fileNumber']      = jsn['filings']['recent']['fileNumber']
        #df['filmNumber']      = jsn['filings']['recent']['filmNumber']
        #df['primaryDocument'] = jsn['filings']['recent']['primaryDocument']
        #
        df = df[df['form'] == arg_doc_type]    # narrow down to forms specified as in the argument arg_doc_type
        #
        # URLs for 10-K form documents; Please note other forms might have a different rule for URLs.
        # In the case of a 10-K form document with accessionNumber = '0001018724-21-000004', it goes like this.
        #https://www.sec.gov/Archives/edgar/data/1018724/000101872421000004/0001018724-21-000004.txt    or
        #https://www.sec.gov/Archives/edgar/data/0001018724/000101872421000004/0001018724-21-000004.txt
        #print('https://www.sec.gov/Archives/edgar/data/' + str(CIKs[l]) + '/' + df['accessionNumber'].str.replace('-', '') + '/' + df['accessionNumber'] + '.txt')
        df['URL'] = 'https://www.sec.gov/Archives/edgar/data/' + str(CIKs[l]) + '/' + df['accessionNumber'].str.replace('-', '') + '/' + df['accessionNumber'] + '.txt'
        #
        #print(df.head)
        #
        #print(df.columns)
        df.to_csv(str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.csv', index=False)
        

