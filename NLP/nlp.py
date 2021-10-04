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
# https://github.com/yoshisatoh/Data_Pre-processing/tree/main/NLP/nlp.py
# https://github.com/yoshisatoh/Data_Pre-processing/blob/main/NLP/nlp.py
#
#
########## Input Data File(s)
#
#text_file.txt
#
#
########## Usage Instructions
#
#Run this script on Terminal of MacOS (or Command Prompt of Windows) as follows:
#
#python nlp.py text_file.txt 20
#
#Generally,
#python nlp.py (text file) (n_most_common: number of most common words in the section 9. Making a Frequency Distribution)
#
#
########## Output Data File(s)
#
#text_file_sentences.txt
#text_file_words.txt
#text_file_words_without_stop_words.txt
#text_file_words_without_stop_words_lemmatized.txt
#text_file_words_without_stop_words_lemmatized_stemmed.txt
#text_file_words_without_stop_words_lemmatized_stemmed_freq.most_common.csv
#text_file_words_without_stop_words_stemmed.txt
#text_file_words_without_stop_words_stemmed_tagged.txt
#tree.chinking.ps
#tree.chunking.ps
#tree.NER.NE.ps
#tree.NER.PERSON.ps
#tree.ps
#
#
########## References
#
#Natural Language Processing With Python's NLTK Package
#https://realpython.com/nltk-nlp-python/
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




########## import Python libraries

import sys

import nltk

#1. Tokenizing
from nltk.tokenize import sent_tokenize, word_tokenize
#nltk.download('punkt')    #You might have to run this first.

# 2. Filtering Stop Words
from nltk.corpus import stopwords
#nltk.download("stopwords")    #You might have to run this first.

#3. Stemming
from nltk.stem import PorterStemmer

# 4. Tagging Parts of Speech
#nltk.download('averaged_perceptron_tagger')    #You might have to run this first.

# 5. Lemmatizing
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')    #You might have to run this first.

# 6. Chunking
# 7. Chinking
#nltk.download("averaged_perceptron_tagger")    #You might have to run this first.
from nltk.draw.tree import TreeView

# 8. Named Entity Recognition (NER)
#nltk.download("maxent_ne_chunker")    #You might have to run this first.
#nltk.download("words")    #You might have to run this first.

# 9. Making a Frequency Distribution
from nltk import FreqDist




########## arguments

for i in range(len(sys.argv)):
    print(str(sys.argv[i]))

#print(sys.argv[0])    #nlp_1_prep.py

text_file_name = str(sys.argv[1])    #example_string.txt
n_most_common  = int(sys.argv[2])    #20



########## Load a text file

#open text file in read mode
text_file = open(text_file_name, "r")
#
#read whole file to a string
text_data = text_file.read()
#
#close file
text_file.close()
print(text_file.closed)
#
print(text_data)


#number of lines
text_data_num_lines = sum(1 for line in open(text_file_name, "r"))
#text_file.close()
#print(text_file.closed)
print(text_data_num_lines)




########## 1. Tokenizing
#
# By tokenizing, you can conveniently split up text by word or by sentence. 
# It’s your first step in turning unstructured data into structured data, which is easier to analyze.




##### Tokenizing by sentences
#sent_tokenize()

#sent_tokenize(text_data)
print(sent_tokenize(text_data))

text_file_sentences = sent_tokenize(text_data)
print(text_file_sentences)
#print(type(text_file_sentences))
#<class 'list'>
#print(len(text_file_sentences))

'''
for i in range(len(text_file_sentences)):
    print('line ' + str(i) + ':')
    print(text_file_sentences[i])
'''

# adding a line feed code to each element (word)
obj = map(lambda x: x + "\n", text_file_sentences)
# save as a text file
with open('text_file_sentences.txt', "w", encoding="utf-8") as f:
	f.writelines(obj)
f.close()
print(f.closed)




##### Tokenizing by words
#word_tokenize()

print(word_tokenize(text_data))

text_file_words = word_tokenize(text_data)
print(text_file_words)
#print(type(text_file_words))
#<class 'list'>
#print(len(text_file_words))

'''
for i in range(len(text_file_words)):
    print('word ' + str(i) + ':')
    print(text_file_words[i])
'''

# adding a line feed code to each element (word)
obj = map(lambda x: x + "\n", text_file_words)
# save as a text file
with open('text_file_words.txt', "w", encoding="utf-8") as f:
	f.writelines(obj)
f.close()
print(f.closed)

# Note that "It's" was split at the apostrophe to give you 'It' and "'s", but "Muad'Dib" was left whole.
# This is because NLTK knows that 'It' and "'s" (a contraction of “is”) are two distinct words while "Muad'Dib" is NOT.




########## 2. Filtering Stop Words

#Stop words are words that you want to ignore, so you filter them out of your text when you’re processing it. 
#Very common words like 'in', 'is', and 'an' are often used as stop words since they don’t add a lot of meaning to a text in and of themselves.

#if you need to focus on stop words in "english", then:
stop_words = set(stopwords.words("english"))
print('Stop words')
print(stop_words)
text_file_words_without_stop_words = []

#print(text_file_words)

for word in text_file_words:
    if word.casefold() not in stop_words:
        text_file_words_without_stop_words.append(word)
#
print(text_file_words_without_stop_words)

# adding a line feed code to each element (word)
obj = map(lambda x: x + "\n", text_file_words_without_stop_words)
# save as a text file
with open('text_file_words_without_stop_words.txt', "w", encoding="utf-8") as f:
	f.writelines(obj)
f.close()
print(f.closed)

'''
Content words give you information about the topics covered in the text or the sentiment that the author has about those topics.

Context words give you information about writing style. You can observe patterns in how authors use context words in order to quantify their writing style. Once you’ve quantified their writing style, you can analyze a text written by an unknown author to see how closely it follows a particular writing style so you can try to identify who the author is.

For instance, 'I' is a pronoun, which is part of context words rather than content words.
'not' is technically an adverb but has still been included in NLTK’s list of stop words for English. If you want to edit the list of stop words to exclude 'not' or make other changes, then you can download it.
https://www.nltk.org/nltk_data/
'''




########## 3. Stemming

#Stemming is a text processing task in which you reduce words to their root, which is the core part of a word. For example, the words “helping” and “helper” share the root “help.”
#Stemming allows you to zero in on the basic meaning of a word rather than all the details of how it’s being used. NLTK has more than one stemmer, but you’ll be using the Porter stemmer.

print(type(text_file_words_without_stop_words))

stemmer = PorterStemmer()
#
text_file_words_without_stop_words_stemmed = [stemmer.stem(word) for word in text_file_words_without_stop_words]
#
print(text_file_words_without_stop_words_stemmed)

# adding a line feed code to each element (word)
obj = map(lambda x: x + "\n", text_file_words_without_stop_words_stemmed)
# save as a text file
with open('text_file_words_without_stop_words_stemmed.txt', "w", encoding="utf-8") as f:
	f.writelines(obj)
f.close()
print(f.closed)




########## 4. Tagging Parts of Speech

text_file_words_without_stop_words_stemmed_tagged = nltk.pos_tag(text_file_words_without_stop_words_stemmed)
print(nltk.pos_tag(text_file_words_without_stop_words_stemmed))
#print(type(nltk.pos_tag(text_file_words_without_stop_words_stemmed)))
#<class 'list'>

#nltk.help.upenn_tagset()

'''
Tags that start with	Deal with
JJ	Adjectives
NN	Nouns
RB	Adverbs
PRP	Pronouns
VB	Verbs
'''

# adding a line feed code to each element (word)
'''
obj = map(lambda x: x + "\n", text_file_words_without_stop_words_stemmed_tagged)
# save as a text file
with open('text_file_words_without_stop_words_stemmed_tagged.txt', "w", encoding="utf-8") as f:
	f.writelines(obj)
f.close()
print(f.closed)
'''
f = open('text_file_words_without_stop_words_stemmed_tagged.txt', 'w')
for t in text_file_words_without_stop_words_stemmed_tagged:
    line = ', '.join(str(x) for x in t)
    f.write(line + '\n')
f.close()
print(f.closed)




########## 5. Lemmatizing
#Like stemming, lemmatizing reduces words to their core meaning, but it will give you a complete English word that makes sense on its own instead of just a fragment of a word like 'discoveri'.

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("scarves"))
#'scarf'

print(lemmatizer.lemmatize("worst"))    #worst as a noun
#'worst'

print(lemmatizer.lemmatize("worst", pos="a"))    #worst as an adjective
#'bad'


##### Lemmatizing
text_file_words_without_stop_words_lemmatized = [lemmatizer.lemmatize(word) for word in text_file_words_without_stop_words]
#print(text_file_words_without_stop_words)
#print(text_file_words_without_stop_words_lemmatized)

# adding a line feed code to each element (word)
obj = map(lambda x: x + "\n", text_file_words_without_stop_words_lemmatized)
# save as a text file
with open('text_file_words_without_stop_words_lemmatized.txt', "w", encoding="utf-8") as f:
	f.writelines(obj)
f.close()
print(f.closed)


##### Stemming (after Lemmatizing)
text_file_words_without_stop_words_lemmatized_stemmed = [stemmer.stem(word) for word in text_file_words_without_stop_words_lemmatized]

# adding a line feed code to each element (word)
obj = map(lambda x: x + "\n", text_file_words_without_stop_words_lemmatized_stemmed)
# save as a text file
with open('text_file_words_without_stop_words_lemmatized_stemmed.txt', "w", encoding="utf-8") as f:
	f.writelines(obj)
f.close()
print(f.closed)




########## 6. Chunking

#While tokenizing allows you to identify words and sentences, chunking allows you to identify phrases.

#Chunks don’t overlap, so one instance of a word can be in only one chunk at a time.

#Before you can chunk, you need to make sure that the parts of speech in your text are tagged, so create a string for POS tagging. 

print(text_file_words_without_stop_words_stemmed_tagged)
print(type(text_file_words_without_stop_words_stemmed_tagged))

#You’ve got a list of tuples of all the words in the quote, along with their POS tag. In order to chunk, you first need to define a chunk grammar.


#Create a chunk grammar with one regular expression rule:
#
grammar = "NP: {<DT>?<JJ>*<NN>}"
#
#NP stands for noun phrase.
'''
According to the rule you created, your chunks:

Start with an optional (?) determiner ('DT')
Can have any number (*) of adjectives (JJ)
End with a noun (<NN>)
'''

# draw a tree and save as a ps (PostScript) file

chunk_parser = nltk.RegexpParser(grammar)

tree = chunk_parser.parse(text_file_words_without_stop_words_stemmed_tagged)

tree.draw()

TreeView(tree)._cframe.print_to_file('tree.chunking.ps')

'''
import os
os.system('convert output.ps output.png')
'''
'''
from PIL import Image
Image.open("tree.ps").save("tree.png")
'''



########## 7. Chinking

#Chinking is used together with chunking, but while chunking is used to include a pattern, chinking is used to EXCLUDE a pattern.

#grammer is only one difference between this Chinking and previous Chunking
grammar = """
          Chunk: {<.*>+}
          }<JJ>{"""

chunk_parser = nltk.RegexpParser(grammar)

tree = chunk_parser.parse(text_file_words_without_stop_words_stemmed_tagged)

tree.draw()

TreeView(tree)._cframe.print_to_file('tree.chinking.ps')




########## 8. Named Entity Recognition (NER)

'''
Named entities are noun phrases that refer to specific locations, people, organizations, and so on. With named entity recognition, you can find the named entities in your texts and also determine what kind of named entity they are.
'''

tree = nltk.ne_chunk(text_file_words_without_stop_words_stemmed_tagged)
tree.draw()
TreeView(tree)._cframe.print_to_file('tree.NER.PERSON.ps')


#binary=True if you just want to know what the named entities are but not what kind of named entity they are:

tree = nltk.ne_chunk(text_file_words_without_stop_words_stemmed_tagged, binary=True)
tree.draw()
TreeView(tree)._cframe.print_to_file('tree.NER.NE.ps')




def extract_ne(txt):
    words = word_tokenize(txt, language='english')
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )

#extract_ne(text_data)
print(extract_ne(text_data))




########## 9. Making a Frequency Distribution

text_file_words_without_stop_words_lemmatized_stemmed_freq = FreqDist(text_file_words_without_stop_words_lemmatized_stemmed)
print(text_file_words_without_stop_words_lemmatized_stemmed_freq)
print(text_file_words_without_stop_words_lemmatized_stemmed_freq.most_common(n_most_common))
print(type(text_file_words_without_stop_words_lemmatized_stemmed_freq.most_common(n_most_common)))
text_file_words_without_stop_words_lemmatized_stemmed_freq.most_common = text_file_words_without_stop_words_lemmatized_stemmed_freq.most_common(n_most_common)

#text_file_words_without_stop_words_lemmatized_stemmed_freq.plot(n_most_common, cumulative=True)

with open('text_file_words_without_stop_words_lemmatized_stemmed_freq.most_common.csv', 'w') as f:
    for row in text_file_words_without_stop_words_lemmatized_stemmed_freq.most_common:
        print(*row, sep=',', file=f)
f.close()
print(f.closed)




########## 10. Finding Collocations
#
#A collocation is a sequence of words that shows up often. 

nltk.Text(text_file_words_without_stop_words_lemmatized_stemmed).collocations()