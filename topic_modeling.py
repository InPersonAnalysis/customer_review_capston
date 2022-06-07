# Import for data wrangling
import pandas as pd
import numpy as np
import wrangle
import os

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def acquire_topics(positive_review_series, negative_review_series, ngram_min = 3, ngram_max = 3, n_topics = 5, learning_decay = 0.7, use_cache = True):
    
    # Assign csv name to variable
    postopicfile = 'positive_dominant_topic.csv'
    poswordfile = 'positive_topic_keywords.csv'
    negtopicfile = 'negative_dominant_topic.csv'
    negwordfile = 'negative_topic_keywords.csv'
    
    # Check if files exist and if user wants a fresh copy from the database
    if (os.path.exists(postopicfile) and os.path.exists(poswordfile) and
        os.path.exists(negtopicfile) and os.path.exists(negwordfile) and use_cache):
        print('Using cached csv file...')
        return pd.read_csv(postopicfile), pd.read_csv(poswordfile), pd.read_csv(negtopicfile), pd.read_csv(negwordfile)

    print('Getting a fresh copy')
    # Pull list of Postive and Negative values
    positive_data = positive_review_series.tolist()
    negative_data = negative_review_series.tolist()

    # Create Vectorizer models
    positive_vectorizer = CountVectorizer(min_df=10,
                                 stop_words='english',
                                 token_pattern='[a-zA-Z0-9]{3,}',
                                 ngram_range = (ngram_min,ngram_max))
    negative_vectorizer = CountVectorizer(min_df=10,
                                 stop_words='english',
                                 token_pattern='[a-zA-Z0-9]{3,}',
                                 ngram_range = (ngram_min,ngram_max))

    # Fit and transform vectorizer
    positive_data_vectorized = positive_vectorizer.fit_transform(positive_data)
    negative_data_vectorized = negative_vectorizer.fit_transform(negative_data)

    # Build LDA Model
    positive_lda_model = LatentDirichletAllocation(n_components = n_topics,
                                                   learning_decay = learning_decay,
                                                   learning_method='online',   
                                                   random_state=172,
                                                   n_jobs = -1)
    negative_lda_model = LatentDirichletAllocation(n_components = n_topics,
                                                   learning_decay = learning_decay,
                                                   learning_method='online',   
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
    def show_topics(vectorizer, lda_model, n_words):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    # Get top keywords for each topic
    positive_topic_keywords = show_topics(positive_vectorizer, positive_lda_model, 15)
    negative_topic_keywords = show_topics(negative_vectorizer, negative_lda_model, 15)     

    # Topic - Keywords Dataframe
    positive_df_topic_keywords = pd.DataFrame(positive_topic_keywords)
    positive_df_topic_keywords.columns = ['Word '+str(i) for i in range(positive_df_topic_keywords.shape[1])]
    positive_df_topic_keywords.index = ['Topic '+str(i) for i in range(positive_df_topic_keywords.shape[0])]
    negative_df_topic_keywords = pd.DataFrame(negative_topic_keywords)
    negative_df_topic_keywords.columns = ['Word '+str(i) for i in range(negative_df_topic_keywords.shape[1])]
    negative_df_topic_keywords.index = ['Topic '+str(i) for i in range(negative_df_topic_keywords.shape[0])]

    # Create topic keyword csv
    positive_df_topic_keywords.to_csv('positive_topic_keywords.csv')
    negative_df_topic_keywords.to_csv('negative_topic_keywords.csv')
    
    return positive_dom_top, positive_topic_keywords, negative_dom_top, negative_topic_keywords