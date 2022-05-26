import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

## Kaggle API Command to download
# kaggle datasets download -d jiashenliu/515k-hotel-reviews-data-in-europe

def wrangle_hotel(df):
    '''
    Wrangle Start
    '''
    
    # lower case column names
    df.columns = [col.lower() for col in df]
    
    # Set the review date as a datetime object then set it as the index
    df.review_date = pd.to_datetime(df.review_date)
    df = df.set_index('review_date').sort_index()
    
    # Create columns for date types to groupby
    df['month'] = df.index.month_name()
    df['year'] = df.index.year
    df['day_name'] = df.index.day_name()
    df['day'] = df.index.day
    df['quarter'] = df.index.quarter
    
    # Unique word counts for positive and negative reviews
    df['negative_unique_word_count'] = [len(set(nr.split())) for nr in df.negative_review]
    df['positive_unique_word_count'] = [len(set(pr.split())) for pr in df.positive_review]
    
    # remove day string and make int type
    df.days_since_review = [row.split()[0] for row in df.days_since_review]
    df.days_since_review = df.days_since_review.astype('int')
    
    # Get Hotel Location
    df['location'] = [' '.join(col.split()[-2:]) for col in df.hotel_address]
    
    # Break out tags into groups
    df.tags = [[tag.strip().lower() for tag in (tags.replace('"','').replace("'","")
                                     .replace('[','') .replace(']','')
                                     .split(','))] for tags in df.tags]
    
    return df

# NLP PREPERATION
import nltk
import unicodedata
import re
from nltk.corpus import stopwords

def basic_clean(string):
    '''
    This function takes in a string and
    returns the string normalized.
    '''
    string = unicodedata.normalize('NFKD', string)\
             .encode('ascii', 'ignore')\
             .decode('utf-8', 'ignore')
    string = re.sub(r'[^\w\s]', '', string).lower()

    return string

def tokenize(string):
    '''
    This function takes in a string and
    returns a tokenized string.
    '''
    # Create tokenizer.
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    # Use tokenizer
    string = tokenizer.tokenize(string, return_str = True)

    return string

def stem(string):
    '''
    This function takes in a string and
    returns a string with words stemmed.
    '''
    # Create porter stemmer.
    ps = nltk.porter.PorterStemmer()
    
    # Use the stemmer to stem each word in the list of words we created by using split.
    stems = [ps.stem(word) for word in string.split()]
    
    # Join our lists of words into a string again and assign to a variable.
    string = ' '.join(stems)
    
    return string

def lemmatize(string):
    '''
    This function takes in string for and
    returns a string with words lemmatized.
    '''
    # Create the lemmatizer.
    wnl = nltk.stem.WordNetLemmatizer()
    
    # Use the lemmatizer on each word in the list of words we created by using split.
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    
    # Join our list of words into a string again and assign to a variable.
    string = ' '.join(lemmas)
    
    return string

from nltk.corpus import stopwords

def remove_stopwords(string, extra_words = [], exclude_words = []):
    '''
    This function takes in a string, optional extra_words and exclude_words parameters
    with default empty lists and returns a string.
    '''
    # Create stopword_list.
    stopword_list = stopwords.words('english')
    
    # Remove 'exclude_words' from stopword_list to keep these in my text.
    stopword_list = set(stopword_list) - set(exclude_words)
    
    # Add in 'extra_words' to stopword_list.
    stopword_list = stopword_list.union(set(extra_words))

    # Split words in string.
    words = string.split()
    
    # Create a list of words from my string with stopwords removed and assign to variable.
    filtered_words = [word for word in words if word not in stopword_list]
    
    # Join words in the list back into strings and assign to a variable.
    string_without_stopwords = ' '.join(filtered_words)
    
    return string_without_stopwords
# One and done fuction for NLP
def nlp_clean(df):
    '''
    This function takes in a Customer Review Data and returns NLP prep.
    '''
    # Apply basic clean and tokenize to each review.
    df['positive_clean_review'] = df['positive_review'].apply(basic_clean)
    df['negative_clean_review'] = df['negative_review'].apply(basic_clean)

    df['positive_clean_review'] = df['positive_clean_review'].apply(tokenize)
    df['negative_clean_review'] = df['negative_clean_review'].apply(tokenize)

    df['positive_clean_review'] = df['positive_clean_review'].apply(remove_stopwords)
    df['negative_clean_review'] = df['negative_clean_review'].apply(remove_stopwords)
    # Apply stem to each review.
    df['positive_stem'] = [stem(review) for review in df.positive_clean_review]
    df['negative_stem'] = [stem(review) for review in df.negative_clean_review]
    # Apply lemmatize to each review.
    df['positive_lemma'] = [lemmatize(review) for review in df.positive_clean_review]
    df['negative_lemma'] = [lemmatize(review) for review in df.negative_clean_review]
    return df
#### END NLP PREPERATION