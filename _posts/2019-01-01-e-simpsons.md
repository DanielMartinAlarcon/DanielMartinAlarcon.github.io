---
title: Simpson Says
subtitle: Natural Language Processing and web app deployment
image: /img/8_simpson/homer.png
date: 2019-01-01 01:46:00
---

I used simple NLP to write a search function that will compare a user query to each line of dialogue in ~600 episodes of The Simpsons, and deployed it in a live web application. This was a one-week team project with other data scientists and web developers at Lambda School, and resulted in [a fully functional website](https://simpsonssays.netlify.com/) where you can access our models yourself. The full repo for this project is [here](https://github.com/simpson-says/buildweek3-simpsons-says-ds).

# Using text similarity to find iconic quotes.
The full script for about 27 seasons and 600 episodes of the show are available [on Kaggle](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data). A relatively small amount of processing is needed to create a semantic index of all the existing quotes, to which new quotes can be compared for similarities.  We did that here using the Natural Language Toolkit ([NLTK](https://www.nltk.org/)) and the semantic analysis tool [Gensim](https://radimrehurek.com/gensim/). 

The basic steps for text processing are:  

1. **Clean the text.**  I made a pandas dataframe where the column `normalized_text` contains all the lines of dialog for all the episodes, stripped of punctuation and turned to lowercase.  
2. **Tokenize.** Each row was a string with a single line of dialog.  I split that string into a list of individual words. 
3. **Create a dictionary.** I mapped each word to a single number, in a single dictionary.
4. **Represent each line as a bag-of-words.** Turn each line into a vector of the same length as the dictionary, with a number for every time that a word occurs in the sentence (since the dictionary is much more vast than the sentence, the vector will mostly be composed of zeroes).
5. **Calculate the [TF-IDF score](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for each word.** The TF-IDF score represents how important a word is in the corpus overall. It is higher for words that appear many times in the document, and lower for words that appear a lot in the entire corpus.
6. **Generate a similarity index.** Gensim creates a semantic index based on cosine similarity that can be queried very efficiently.  Thi index is a large file, which gets split into several shards that must be loaded into memory for a new query.

The most concise form of the code looks like this:

```python
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

df2 = pd.read_csv('simpsons_script_lines.csv',error_bad_lines=False)
df2['token'] = [word_tokenize(str(x)) for x in df2['normalized_text']]
dictionary = gensim.corpora.Dictionary(df2['token'])
corpus = [dictionary.doc2bow(row) for row in df2['token']]
tf_idf = gensim.models.TfidfModel(corpus)
similarity_index = gensim.similarities.Similarity('/shards/shard',tf_idf[corpus],
                                      num_features=len(dictionary))
```
And that's it!  New queries from the user are processed in the same manner as all the original lines of dialog, and queried against the similarity index.  The result is an array of how similar the query is to *every single line of dialog ever*.  We query the original dataframe for the top 10 most similar indices in that array, and thus get the top 10 quotes that most closely match the query.

```python
query_tokenized = [w.lower() for w in word_tokenize(query)]
query_bow = dictionary.doc2bow(query_tokenized)
query_tfidf = tf_idf[query_bow]
array_of_similarities = similarity_index[query_tfidf]
top_10_indices = array_of_similarities.argsort()[-10:][::-1]
top_10_lines = df[df.index.isin(result)]
```
We wrote up a web app to return raw results that look something like this:

```json
[{"quote_id":13144,"raw_character_text":"Bart Simpson","spoken_words":"Hey, Homer, I can't find the safety goggles for the power saw.","episode_title":"Saturdays of Thunder","season":3,"number_in_season":9},
{"quote_id":37513,"raw_character_text":"Radioactive Man","spoken_words":"My eyes! The goggles do nothing.","episode_title":"Radioactive Man","season":7,"number_in_season":2},
{"quote_id":64870,"raw_character_text":"Milhouse Van Houten","spoken_words":"My sport goggles!","episode_title":"Brother's Little Helper","season":11,"number_in_season":2},
{"quote_id":78425,"raw_character_text":"Marge Simpson","spoken_words":"I want goggles, too.","episode_title":"The Parent Rap","season":13,"number_in_season":2},]
```

Our team of web developers turned this into a full website that also allows users to log in and save favorite quotes:

![Squid query](/img/8_simpson/baltic1.png)

![Squid results](/img/8_simpson/baltic2.png)

[You can try it yourself here](https://simpsonssays.netlify.com/).  Note that, because we are using a free tier of Heroku to deploy the app, the website takes about 2 minutes to load results when it hasn't been accessed for a while.

# Using a recurrent neural network to generate new text.
My colleage adapted the TensorFlow code from [this blog post](https://towardsdatascience.com/how-to-generate-your-own-the-simpsons-tv-script-using-deep-learning-980337173796) to build a recurrent neural network (RNN), train it with dialog lines from specific characters, and generate synthetic quotes that we also fed into our website. The full work can be found in [this Jupyter Notebook](https://github.com/simpson-says/buildweek3-simpsons-says-ds/blob/master/Simpsons_Writes_V4.ipynb).

We trained the RNN on dialog lines from several characters, then collected the results into a single dictionary with ~100 quotes per character (after removing really short quotes because they don't really capture any specific style).  Here are some sample synthetic lines for Marge Simpson:

```
(reading) you can't have a lot of mittens this is
(indignant) you promised to know what about the gun?
(angry) bart was where we ever just?
lisa! i just think you're spoiling them.
```

# Deploying the model to the web
In order to make the results of our models available online, we created a  web application using [Flask](http://flask.pocoo.org/) that our web developers could connect to.  The application itself is deployed on the cloud platform Heroku, and responds to user POST requests with JSON objects containing the results of our model.  Here is a cleaned up version of the app, commented to explain functionality.


First, import all dependencies and many of the elements of the search function described above, which have been pickled before and must be loaded here for processing new queries.
```python
from flask import Flask , request, make_response
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import gensim
import random
import json

APP = Flask(__name__)

### Recover pickled elements
# The dataframe of quotes
quote_list = pickle.load( open( "quote_list.pkl", "rb" ) )
# Dictionary of each word to a number
dictionary = pickle.load( open( "dictionary.pkl", "rb" ) )
# The bag-of-words representation of all the dialog lines
corpus = pickle.load( open( "corpus.pkl", "rb" ) )
# TF-IDF value of each word in the previous
tf_idf = pickle.load( open( "tf_idf.pkl", "rb" ) )
# The similarity index
similarity_index = pickle.load( open( "similarity_index.pkl", "rb" ) )
```

This flask app contains several routes for the different functionalities of the website.  The first one is the `/api` route, which allows our web team to send a user query and retrieve a JSON file of related quotes from the show.
```python
@APP.route('/')
@APP.route('/api', methods=['POST'])
def api():

    # The POST request contains a user's query.  This replaces an existing
    # quote that we keep as a placeholder so this route works even in the 
    # absence of a POST request.
    user_input = "the goggles do nothing"
    if request.method == 'POST':
        user_input = request.values['quote']

    # Tokenize the words in the user input, removing punctuation
    query_doc = [w.lower() for w in word_tokenize(user_input)]
    # Turn into a bag-of-words
    query_doc_bow = dictionary.doc2bow(query_doc)
    # Calculate the TF_IDF of the query
    query_doc_tf_idf = tf_idf[query_doc_bow]
    # Query the similarity index
    array_of_similarities = similarity_index[query_doc_tf_idf]
    # Extract the top 10 closest results
    top_10_indices = array_of_similarities.argsort()[-10:][::-1]
    # Return the original show quotes that match the indices of the results
    top_10_lines = (quote_list.index.isin(top_10_indices))
    response = quote_list[top_10_lines]
    # Report the following columns of metadata
    column = ['quote_id', 'raw_character_text', 'spoken_words','episode_title','season','number_in_season']
    response = response[column]
    # Transform into a JSON dictionary
    response.to_json(orient='records')
    # Export the results
    return response.to_json(orient='records')

```
The second important route returns synthetic quotes generated by our RNN.  We generated those quotes ahead of time, and exported them as a dictionary with ~100 synthetic quotes for a handful of characters. This route takes in a character name (from a short pre-determined list) and returns a random sampling of 10 synthetic quotes.
```python
# Pre-generated quotes
syn_quote_list = pickle.load(open("syn_quote_list.pkl", "rb" ))

@APP.route('/gen', methods=['POST'])
@APP.route('/gen')
def generator():
    # Acceptable inputs = ['homer', 'marge', 'bart', 'lisa', 'moe', 'grampa', 'skinner']
    
    # A placehorder query gets replaced with the contents of a POST request.
    name = 'homer'
    if request.method=='POST':
        name = request.values['input']
    
    # Select and return a quote by that character.
    rand_quotes = random.choices(syn_quote_list[name], k=10)
    quotes = [{'charname':name, 'quote':x} for x in rand_quotes]
    return json.dumps(quotes)
```
We also built a third route for returning a particular quote from the original list in response to a number.  This was important so that the web team could allow users to save favorite quotes. 

You can try out our barebones web application [here](https://eat-my-shorts.herokuapp.com/), and imagine how our team of web developers transformed that into [the full website](https://simpsonssays.netlify.com/).  You can find the full code of our web application [here](https://github.com/simpson-says/buildweek3-simpsons-says-ds/blob/master/app.py).