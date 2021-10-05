#################### Data Pre-processing: Natural language processing (NLP) for 10-k fillings ####################
#
#  (C) 2021, Yoshimasa (Yoshi) Satoh, CFA 
#
#  All rights reserved.
#
# Created:      2021/10/05
# Last Updated: 2021/10/05
#
# Github:
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/sec-edgar/sec-edgar.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/sec-edgar/sec-edgar.py
#
#
########## Input Data File(s)
#
#tickers_CIKs.txt    #one column only
#filing_types.txt    #one column only
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#python sec-edgar.py tickers_CIKs.txt filing_types.txt 2017-01-01 2017-03-31
#
#Generally,
#python sec-edgar.py (a text file of tickers and/or CIKs) (a text file of filing types) (arg_date_after), (arg_date_before)
#
#
########## Output Data File(s)
#
#
#
#
########## References
#
#EDGAR Company Filings | CIK Lookup
#The Central Index Key (CIK) is used on the SEC's computer systems to identify corporations and individual people who have filed disclosure with the SEC.
#https://www.sec.gov/edgar/searchedgar/cik.htm
#
#sec-edgar-downloader 4.2.2
#https://pypi.org/project/sec-edgar-downloader/
#
#Natural Language Toolkit (NLTK)
#https://www.nltk.org/
#
#NLP in the Stock Market
#https://towardsdatascience.com/nlp-in-the-stock-market-8760d062eb92
#https://github.com/roshan-adusumilli/nlp_10-ks
#Note that you might have difficulty in running scripts on these websites. Thus, I made up my mind to build my own from scratch.
#
#
####################




########## Background: SEC EDGAR database and 10-k forms
#
#SEC EDGAR database provides various forms of filings.
#10-k forms are annual reports filed by companies to provide a comprehensive summary of their financial performance (these reports are mandated by the Securities and Exchange Commission). 
#Combing through these reports is often tedious for investors.
#Through sentiment analysis, a subfield of natural language processing, investors can quickly understand if the tone of the report is positive, negative, or litigious etc.
#The overall sentiment expressed in the 10-k form can then be used to help investors decide if they should invest in the company.




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
#pip install --upgrade sec-edgar-downloader --trusted-host pypi.org --trusted-host files.pythonhosted.org




########## import Python libraries

import sys
import csv

import nltk

import numpy as np
import pandas as pd

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import warnings
warnings.simplefilter('ignore')


from sec_edgar_downloader import Downloader




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #sec-edgar.py

arg_tickers_CIKs_fname = str(sys.argv[1])    #tickers_CIKs.txt

arg_filing_types_fname = str(sys.argv[2])    #filing_types.txt

arg_date_after         = str(sys.argv[3])    #2017-01-01
arg_date_before        = str(sys.argv[4])    #2017-03-31




########## load input files


with open(arg_tickers_CIKs_fname, 'r') as f:
    list_tickers_CIKs = f.read().splitlines()
#
print(list_tickers_CIKs)
#print(f.closed)


with open(arg_filing_types_fname, 'r') as f:
    list_filing_types = f.read().splitlines()
#
print(list_filing_types)
#print(f.closed)




########## sec-edgar-downloader

# Download all the specified filings of all tickers to the current working directory
dl = Downloader()




for ticker_CIK in list_tickers_CIKs:
    for filing_type in dl.supported_filings:
        #
        dl.get(filing_type, ticker_CIK, after=arg_date_after, before=arg_date_before)
        #dl.get(filing_type, ticker_CIK, after="2017-01-01", before="2017-03-25", verify=False)

