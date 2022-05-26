import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

## Kaggle API Command to download
# kaggle datasets download -d jiashenliu/515k-hotel-reviews-data-in-europe

def parse_tags(tags):
    tags = tags.lower()
    trip_type = 'unknown'
    if 'leisure trip' in tags:
        trip_type = 'leisure'
    elif 'buisness trip' in tags:
        trip_type = 'business'
    nights_stayed = np.nan
    if re.search(r'stayed\s*(\d+)\s*nights?', tags):
        nights_stayed = re.sub(r'.*stayed\s*(\d+)\s*nights?.*', r'\1', tags)
    group_type = 'unknown'
    if 'group' in tags:
        group_type = 'group'
    elif 'solo traveler' in tags:
        group_type = 'solo traveler'
    elif 'family with young children' in tags:
        group_type = 'family with young children'
    elif 'family with older children' in tags:
        group_type = 'family with older children'
    elif 'couple' in tags:
        group_type = 'couple'
    elif 'travelers with friends' in tags:
        group_type = 'travelers with friends'
    return dict(trip_type = trip_type, nights_stayed = nights_stayed, group_type = group_type)


def wrangle_hotel(df):
    '''
    Wrangle Start
    '''

    # lower case column names
    df.columns = [col.lower() for col in df]
    
    tags = pd.DataFrame(df.tags.apply(parse_tags).tolist())
    df = pd.concat([df, tags], axis=1)


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
    # Create blank lists
    street = []
    city = []
    zip_code = []
    country = []
    
    # loop through addresses
    for address in df.hotel_address:
        # If France, Netherlands or Italy then split address as follows
        if address.split()[-1] in ['France','Netherlands','Italy']:

            street.append(' '.join(address.split()[:-4]))
            zip_code.append(' '.join(address.split()[-4:-2]))
            city.append(' '.join(address.split()[-2:-1]))
            country.append(' '.join(address.split()[-1:]))
        # If Spain, Austria then split address as follows
        elif address.split()[-1] in ['Spain','Austria']:

            street.append(' '.join(address.split()[:-3]))
            zip_code.append(' '.join(address.split()[-3:-2]))
            city.append(' '.join(address.split()[-2:-1]))
            country.append(' '.join(address.split()[-1:]))
        # United Kindoms is split last
        else:
            street.append(' '.join(address.split()[:-5]))
            city.append(' '.join(address.split()[-5:-4]))
            zip_code.append(' '.join(address.split()[-4:-2]))
            country.append(' '.join(address.split()[-2:]))
    
    # Assign columns
    df['street'] = street
    df['city'] = city
    df['zip_code'] = zip_code
    df['country'] = country
    
    # Drop Address
    df.drop('hotel_address',inplace=True)
    
    return df