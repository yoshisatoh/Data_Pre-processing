#################### Data Pre-processing: Natural language processing (NLP) for SEC EDGAR database API - 1. Getting JSON metadata for each entity with a unique CIK ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/07
# Last Updated: 2021/10/07
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sec-edgar_api_data_nlp/sec-edgar_api_1_get_json.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/sec-edgar_api_data_nlp/sec-edgar_api_1_get_json.py
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
#python sec-edgar_api_1_get_json.py CIKs.txt
#
#Generally,
#python sec-edgar_api_1_get_json.py (arg_CIKs_file_name)
#
#
########## Output Data File(s)
#
#json files as in:
#str(CIKs[l]) + '/CIK' + str(CIKs[l]) + '.json'
#
#For instance,
#0001018724/CIK0001018724.json
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




########## EDGAR Application Programming Interfaces
#
#https://www.sec.gov/edgar/sec-api-documentation
#
'''
Currently included in the APIs are the submissions history by filer and the XBRL data from financial statements (forms 10-Q, 10-K,8-K, 20-F, 40-F, 6-K, and their variants).

The JSON structures are updated throughout the day, in real time, as submissions are disseminated.

Each entity’s current filing history is available at the following URL:

https://data.sec.gov/submissions/CIK##########.json

Where the ########## is the entity’s 10-digit Central Index Key (CIK), including leading zeros.

This JSON data structure contains metadata such as current name, former name, and stock exchanges and ticker symbols of publicly-traded companies. The object’s property path contains at least one year’s of filing or to 1,000 (whichever is more) of the most recent filings in a compact columnar data array. If the entity has additional filings, files will contain an array of additional JSON files and the date range for the filings each one contains.
'''




########## Get all the entitys' json meta data of filings.

# CIKs: A list of CIKs for all the entities.

for l in range(len(CIKs)):
    #
    #Set a URL for a json file.
    url = 'https://data.sec.gov/submissions/CIK'+ str(CIKs[l]) + '.json'
    #
    #Get a json metadata for an entity    
    #j = requests.get(url, allow_redirects=True)    #If you see an error of [SSL: CERTIFICATE_VERIFY_FAILED], try the following:
    j = requests.get(url, allow_redirects=True, verify=False)
    #
    #Create a directory for the entity with a unique CIK - if it already exists, then skip
    if not os.path.exists(str(CIKs[l])):
        os.makedirs(str(CIKs[l]))
    #
    #Save a jason metadata onto the directory with the name of the CIK
    open(str(CIKs[l]) + '/' + str(CIKs[l]) + '.json', 'wb').write(j.content)
    #
    time.sleep(60/10+1)
    #Note that the SEC will limit automated searches to a total of no more than 10 requests per second.
    #https://www.sec.gov/oit/announcement/new-rate-control-limits
    #

print('If you find a json file (e.g., 0001018724.json without CIK prefix) with the following message in the h1 tag, then you reached the limit by the SEC.')
print('<h1>Your Request Originates from an Undeclared Automated Tool</h1>')
print('')
print('If that is the case, then look at CIKs.txt and manually download each json file (e.g., CIK0001018724.json) one by one by using the following URL:')
print('https://data.sec.gov/submissions/CIK##########.json')
print('')
print('If CIK is 0001018724 (Amazon Com Inc), then it goes like this.')
print('https://data.sec.gov/submissions/CIK0001018724.json')
print('')