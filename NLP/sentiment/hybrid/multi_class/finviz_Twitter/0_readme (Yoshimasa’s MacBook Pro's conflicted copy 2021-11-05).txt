(C) 2021, Yoshimasa (Yoshi) Satoh, CFA 

All rights reserved.



ARTIFICIAL INTELLIGENCE IN INVESTMENT MANAGEMENT
STUDY GROUP 2021-2022, PROGRAMMING/CONTINUING EDUCATION COMMITTEE, CFA SOCIETY JAPAN

[B] Special Sessions


1. Define a problem to be solved.
2. Prepare (alternative/big) data and and pre-process it by cleaning and standardizing.
3. Evaluate machine learning algorithms including regularization.
4. Improve the outcome by reviewing 2 and 3.
5. Present final results.




1. Define a problem to be solved.

	Relationships between stock prices and dynamically generated text data of news website and social media.


2. Prepare (alternative/big) data and and pre-process it by cleaning and standardizing.

	2.1. Stock Price:	Yahoo Finance (TSLA at 30-min intervals)
	2.2. News:		finviz
	2.3. Social Media:	Twitter


3. Evaluate machine learning algorithms including regularization.

	3.1. Sentiment Analysis
		Hybrid
			Rule-Based	Textblob	multi-class: positive +1, neutral 0, or negative -1	subjectivity: 0 (objective) to 1 (subjective)
			Rule-Based	Vader Sentiment	multi-class: positive, neutral, or negative (from 0 to 1 for each class)
			Embedding-based	Flair		binary-class: positive or negative (and value)
	(3.2. Sentiment and Stock Price Analysis)
		Compare results in 2.1. and 3.1.
			2.2.2.df_parsed_news.csv
			3.1.5.df_parsed_news__df_textblob_vs_flr.csv


4. Improve the outcome by reviewing 2 and 3.

5. Present final results.




How to run Python scripts: save all the py files on a local directory and then run python on your Command Prompt (Windows) or Terminal (MacOS).

python 2.1.0.yf.py TSLA 2021-09-30 2021-11-01 30m

python 2.2.0.finviz.py 2.2.0.tickers.txt

python 3.1.0.nlpsahbmctxt.py 2.2.2.df_parsed_news.csv
	Save all the output files:
		3.1.1.df_textblob.csv
		3.1.2.df_vs.csv
		3.1.3.df_flr.csv
		3.1.4.df_textblob_vs_flr.csv
		3.1.5.df_parsed_news__df_textblob_vs_flr.csv


python 3.1.0.nlpsahbmctxt.py 2.3.3.tweets.csv
	Save all the output files:
		(See above)

