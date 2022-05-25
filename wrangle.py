import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re


# Kaggle API Command to download

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