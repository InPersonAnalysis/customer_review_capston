import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from nltk.sentiment import SentimentIntensityAnalyzer
import os
import kaggle

def acquire_hotel_data():
    '''
    This function utilizes Kaggle API to acquire hotel data.
    Before running this function, you must have a Kaggle API key. Instructions to obtain key is in readme.md
    '''
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('jiashenliu/515k-hotel-reviews-data-in-europe', path='./' , unzip=True)
    df = pd.read_csv('Hotel_Reviews.csv')
    return df

#function for parsing the tags column in the dataframe
def parse_tags(tags):
    #change all tag values to lower case
    tags = tags.lower()
    #initialize the trip type variable as 'unknown'
    trip_type = 'unknown'
    #parse trip type and pull out trip type values from column
    if 'leisure trip' in tags:
        trip_type = 'leisure'
    elif 'business trip' in tags:
        trip_type = 'business'
    #initialize the nights stayed variable as 'nan'
    nights_stayed = np.nan
    #parse the nights stayed values by pulling out the digit with regex
    if re.search(r'stayed\s*(\d+)\s*nights?', tags):
        nights_stayed = re.sub(r'.*stayed\s*(\d+)\s*nights?.*', r'\1', tags)
    #initialize the group type variable as 'unknown'
    group_type = 'unknown'
    #parse the group type and pull group type values
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
    #return a dictionary with parsed values
    return dict(trip_type = trip_type, nights_stayed = nights_stayed, group_type = group_type)


def wrangle_hotel(use_cache=True):
    '''
    Transform the dataframe so that time, address and tags are all useable. Add NLP data to dataframe.
    '''

    # Assign filename to csv for storage
    filename = 'hotel.json'
    
    # Check if file exists and if user wants a fresh copy from the database
    if os.path.exists(filename) and use_cache:
        print('Using cached file...')
        return pd.read_json(filename)

    df = acquire_hotel_data()
    # lower case column names
    df.columns = [col.lower() for col in df]
    
    #use the parse_tags function to parse the string values in the tags columns and create new feature columns
    tags = pd.DataFrame(df.tags.apply(parse_tags).tolist())
    #Concatenate the new features to the original dataframe
    df = pd.concat([df, tags], axis=1)

    # Set the review date as a datetime object then set it as the index
    df.review_date = pd.to_datetime(df.review_date)
    df = df.set_index('review_date').sort_index()

    # Create columns for date types to groupby
    df['month'] = df.index.month
    df['month_name'] = df.index.month_name()
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
    df['nps_group'] = df.reviewer_score.apply(nps_group)

    # Drop Address
    df = df.drop(columns=['hotel_address','tags'])
    
    # NLP Clean
    df = nlp_clean(df)
    
    # Topic Assignment
    df['positive_topic'] = df.positive_lemma.apply(top_topic)
    df['negative_topic'] = df.negative_lemma.apply(top_topic)
    
    # Rearrange columns
    columns = ['month_name','month','year','day_name','day','quarter','hotel_name','street','city','zip_code','country','lat','lng',
        'additional_number_of_scoring','average_score','total_number_of_reviews','reviewer_nationality','trip_type','nights_stayed',
        'group_type','total_number_of_reviews_reviewer_has_given','reviewer_score','nps_group','days_since_review','neg_sentiment_score',
        'neg_lem_sentiment_score','review_total_negative_word_counts','negative_unique_word_count','negative_topic','pos_sentiment_score',
        'review_total_positive_word_counts','positive_unique_word_count','pos_lem_sentiment_score','positive_topic','negative_review',
        'negative_clean_review','negative_lemma','positive_review','positive_clean_review','positive_lemma']
    
    df = df[columns]
    df = df.reset_index()
    
    # Create csv
    df.to_json(filename, date_format='iso')
    
    return df

def fill_coord_nulls(df):
    # Get median latitude and longitude for each city in df. Assign to variables.
    med_lats = df.groupby('city').lat.agg(['median'])
    med_longs = df.groupby('city').lng.agg(['median'])
    
    # Fill null values in 'lat' column with median latitude coordinate according to city by variable assignment
    amster_lats = df[(df.city == 'Amsterdam')].lat.fillna(med_lats.iloc[0][0])
    barza_lats = df[(df.city == 'Barcelona')].lat.fillna(med_lats.iloc[1][0])
    london_lats = df[(df.city == 'London')].lat.fillna(med_lats.iloc[2][0])
    milan_lats = df[(df.city == 'Milan')].lat.fillna(med_lats.iloc[3][0])
    paris_lats = df[(df.city == 'Paris')].lat.fillna(med_lats.iloc[4][0])
    vienna_lats = df[(df.city == 'Vienna')].lat.fillna(med_lats.iloc[5][0])
    
    # Reassign 'lat' column with concatenated variables to fill nulls with median latitude by city
    df.lat = pd.concat([amster_lats, barza_lats, london_lats, milan_lats, paris_lats, vienna_lats], axis=0, ignore_index=False)

    # Fill null values in 'lng' column with median longitude coordinate according to city by variable assignment
    amster_longs = df[(df.city == 'Amsterdam')].lng.fillna(med_longs.iloc[0][0])
    barza_longs = df[(df.city == 'Barcelona')].lng.fillna(med_longs.iloc[1][0])
    london_longs = df[(df.city == 'London')].lng.fillna(med_longs.iloc[2][0])
    milan_longs = df[(df.city == 'Milan')].lng.fillna(med_longs.iloc[3][0])
    paris_longs = df[(df.city == 'Paris')].lng.fillna(med_longs.iloc[4][0])
    vienna_longs = df[(df.city == 'Vienna')].lng.fillna(med_longs.iloc[5][0])

    # Reassign 'lng' column with concatenated variables to fill nulls with median longitude by city
    df.lng = pd.concat([amster_longs, barza_longs, london_longs, milan_longs, paris_longs, vienna_longs], axis=0, ignore_index=False)
    
    return df

# NLP PREPARATION
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
    string = re.sub(r'[\s]', ' ', string).strip()

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

# One and done function for NLP
def nlp_clean(df):
    '''
    This function takes in a df of Customer Review Data and returns NLP prep.
    '''
    # Apply basic clean, tokenize and remove stopwords to each review.
    df['positive_clean_review'] = df['positive_review'].apply(basic_clean)
    df['negative_clean_review'] = df['negative_review'].apply(basic_clean)

    # Remove 'No Negative' and 'No Positive' from the reviews

    words = ['nothing', 'n', 'none', 'nothing really', 'good', 'nothing dislike', 'liked everything', 'everything perfect', 'nil', 'nothing complain', 'nothing say', 'no negative']
    import re 
    df['positive_clean_review'] = [review.replace('no positive', '') for review in df.positive_clean_review]
    df.loc[df.negative_clean_review.isin(words), 'negative_clean_review'] = ''
    #df['negative_clean_review'] = [re.sub(words, '', review) for review in df.negative_clean_review]

    df.positive_clean_review.fillna('', inplace=True)
    df.negative_clean_review.fillna('', inplace=True)

    df['positive_clean_review'] = df['positive_clean_review'].apply(remove_stopwords)
    df['negative_clean_review'] = df['negative_clean_review'].apply(remove_stopwords)

    df['positive_clean_review'] = df['positive_clean_review'].apply(tokenize)
    df['negative_clean_review'] = df['negative_clean_review'].apply(tokenize)

    # Apply stem to each review.
    #df['positive_stem'] = [stem(review) for review in df.positive_clean_review]
    #df['negative_stem'] = [stem(review) for review in df.negative_clean_review]
    
    # Apply lemmatize to each review.
    df['positive_lemma'] = [lemmatize(review) for review in df.positive_clean_review]
    df['negative_lemma'] = [lemmatize(review) for review in df.negative_clean_review]

    sia = SentimentIntensityAnalyzer()
    df['pos_sentiment_score'] = df.positive_clean_review.apply(lambda msg: sia.polarity_scores(msg)['compound'])
    df['neg_sentiment_score'] = df.negative_clean_review.apply(lambda msg: sia.polarity_scores(msg)['compound'])
    df['pos_lem_sentiment_score'] = df.positive_lemma.apply(lambda msg: sia.polarity_scores(msg)['compound'])
    df['neg_lem_sentiment_score'] = df.negative_lemma.apply(lambda msg: sia.polarity_scores(msg)['compound'])
    
    return df
#### END NLP PREPARATION

## NPS Score Feature Engineering
def nps_group(reviewer_score):
    '''
    This function takes in the reviewer's score and assigns the review a categorical label ('promoter', 'passive', 'detractor')
    '''
    #score of 9 or more = 'promoter'
    if reviewer_score > 8.9:
        nps_group = 'promoter'
    #score between 7 and 9 = 'passive'
    elif reviewer_score > 6.9:
        nps_group = 'passive'
    #score less than 7 = detractor
    elif reviewer_score >= 0:
        nps_group = 'detractor'
    #if no score return no group
    else:
        nps_group = 'no group'
    return nps_group
    
def top_topic(review):
    '''
    This function takes in a review and checks for keywords assigned for each topic.
    It assigns a number value to each topic and returns the topic with the highest count.
    '''
    # Make list of words for each topic
    staff_words = ['staff','friendly','helpful','reception','maids','attentive','concierge','service','receptionist']
    
    location_words = ['location','located','view','area','metro','city','central','centre','center','train','access','walk',
                      'public','transport','transportation','downtown','accessible','station','convenient','shop','distance']
    
    room_words = ['room','bed','matress','bathroom','spacious','shower','suite','pillow','bedroom','conditioning','bathtub','air',
                  'window','conditioned','double','twin','size','single','noise','tv','bath','carpet','tile','noisy','small',
                  'furniture','quiet','toilet']
    
    facilities_words = ['facilities','breakfast','security','buffet','restaurant','bar','coffee','tea','pool','wifi','facility',
                        'parking','lobby','lounge','drink','hall','corridor','menu','storage','order','dining']
    
    value_words = ['value','money','cheap','cheapest','expensive','price','priced','cost']
    
    hotel_words = ['hotel']
    
    comfort_words = ['nothing','everything','atmosphere','good','comfortable','comfort','uncomfortable','pretty','clean','experience']

    # Make a dictionary to check for topic words and a dictionary to keep topic counts per review
    topic_dict = {'staff':staff_words,'location':location_words,'room':room_words,'facilities':facilities_words,
                  'value':value_words,'hotel':hotel_words,'comfort':comfort_words}
    
    topic_count = {'staff' : 0, 'location' : 0, 'room' : 0, 'facilities' : 0, 'value' : 0, 'hotel': 0, 'comfort': 0}
    
    # Check the review for the topics
    if review == '':
        return 'blank'
    
    elif review.isnumeric():
        return 'number'
    
    for word in review.split():
        if word in topic_dict['staff']:
            topic_count['staff'] += 1
        
        elif word in topic_dict['location']:
            topic_count['location'] += 1
        
        elif word in topic_dict['room']:
            topic_count['room'] += 1
        
        elif word in topic_dict['facilities']:
            topic_count['facilities'] += 1
        
        elif word in topic_dict['value']:
            topic_count['value'] += 1
        
        elif word in topic_dict['hotel']:
            topic_count['hotel'] += 1
        
        elif word in topic_dict['comfort']:
            topic_count['comfort'] += 1
    
    if all(value == 0 for value in topic_count.values()):
        return 'no_topic'
    
    else:
        return max(topic_count, key = topic_count.get)

