# Import for data wrangling
import pandas as pd
import numpy as np
import wrangle
import pickle
import os

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

def acquire_topics(df,use_cache = False):
    
    
    topicfile = 'topic_file.csv'
    wordfile = 'word_file.csv'
    
    # Check if file exists and if user wants a fresh copy from the database
    if os.path.exists(topicfile) and os.path.exists(wordfile) and use_cache:
        print('Using cached csv file...')
        return pd.read_csv(topicfile), pd.read_csv(wordfile)

    
    
    # Pull list of Postive and Negative values
    positive_data = hotel.positive_lemma.tolist()
    negative_data = hotel.negative_lemma.tolist()

    # Create Vectorizer models
    postive_vectorizer = CountVectorizer(min_df=10,
                                 stop_words='english',
                                 token_pattern='[a-zA-Z0-9]{3,}')
    negative_vectorizer = CountVectorizer(min_df=10,
                                 stop_words='english',
                                 token_pattern='[a-zA-Z0-9]{3,}')

    # Fit and transform vectorizer
    positive_data_vectorized = vectorizer.fit_transform(positive_data)
    negative_data_vectorized = negative_vectorizer.fit_transform(negative_data)

    # Build LDA Model
    positive_lda_model = LatentDirichletAllocation(learning_method='online',   
                                          random_state=172,
                                          n_jobs = -1)
    negative_lda_model = LatentDirichletAllocation(learning_method='online',   
                                          random_state=172,
                                          n_jobs = -1)

    # Fit and transform lda model
    positive_lda_output = positive_lda_model.fit_transform(positive_data_vectorized)
    negative_lda_output = negative_lda_model.fit_transform(negative_data_vectorized)

    # column names
    positive_topicnames = ["Topic" + str(i) for i in range(positive_lda_model.n_components)]
    negative_topicnames = ["Topic" + str(i) for i in range(negative_lda_model.n_components)]

    # index names
    positive_docnames = ["Doc" + str(i) for i in range(len(positive_data))]
    negative_docnames = ["Doc" + str(i) for i in range(len(negative_data))]

    # Make the pandas dataframe
    positive_df_document_topic = pd.DataFrame(np.round(positive_lda_output, 2), columns=positive_topicnames, index=positive_docnames)
    negative_df_document_topic = pd.DataFrame(np.round(negative_lda_output, 2), columns=negative_topicnames, index=negative_docnames)

    # Get dominant topic for each document
    positive_dominant_topic = np.argmax(positive_df_document_topic.values, axis=1)
    negative_dominant_topic = np.argmax(negative_df_document_topic.values, axis=1)

    # Create Dominant topic csv
    positive_dom_top = pd.DataFrame(positive_dominant_topic)
    positive_dom_top.to_csv('positive_dominant_topic.csv')
    negative_dom_top = pd.DataFrame(negative_dominant_topic)
    negative_dom_top.to_csv('negative_dominant_topic.csv')

    # Topic-Keyword Matrix
    positive_df_topic_keywords = pd.DataFrame(positive_lda_model.components_)
    negative_df_topic_keywords = pd.DataFrame(negative_lda_model.components_)

    # Assign Column and Index
    positive_df_topic_keywords.columns = positive_vectorizer.get_feature_names()
    positive_df_topic_keywords.index = positive_topicnames
    negative_df_topic_keywords.columns = negative_vectorizer.get_feature_names()
    negative_df_topic_keywords.index = negative_topicnames

    # Show top n keywords for each topic
    def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    # Get top keywords for each topic
    positive_topic_keywords = show_topics(vectorizer=positive_vectorizer, lda_model=positive_lda_model, n_words=15)
    negative_topic_keywords = show_topics(vectorizer=negative_vectorizer, lda_model=negative_lda_model, n_words=15)     

    # Topic - Keywords Dataframe
    positive_df_topic_keywords = pd.DataFrame(positive_topic_keywords)
    positive_df_topic_keywords.columns = ['Word '+str(i) for i in range(positive_df_topic_keywords.shape[1])]
    positive_df_topic_keywords.index = ['Topic '+str(i) for i in range(positive_df_topic_keywords.shape[0])]
    negative_df_topic_keywords = pd.DataFrame(negative_topic_keywords)
    negative_df_topic_keywords.columns = ['Word '+str(i) for i in range(negative_df_topic_keywords.shape[1])]
    negative_df_topic_keywords.index = ['Topic '+str(i) for i in range(negative_df_topic_keywords.shape[0])]


    positive_df_topic_keywords.to_csv('positive_topic_keyword.csv')
    negative_df_topic_keywords.to_csv('negative_topic_keywords.csv')
    
    return positive_word_df, positive_topic_df, negative_word_df, negative_topic_df