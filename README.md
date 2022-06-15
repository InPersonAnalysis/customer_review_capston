# Customer Review Analysis

This repository contains all files, and ipython notebooks, used in this project. A full outline of all the files with descriptions can be found below.

To view the Slide Deck, ["click here."](https://www.canva.com/design/DAFC1pAZhjY/37-GQyhTmSVtm3tACkxOEw/edit?utm_content=DAFC1pAZhjY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 

To view the Handout, ["click here."](https://www.canva.com/design/DAFCXzznwGM/qiVjEIAp4Bx0uVw9UfR_Tg/edit?utm_content=DAFCXzznwGM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 

To view the Final Report, ["click here."](https://github.com/InPersonAnalysis/customer_review_capstone/blob/main/final_notebook.ipynb)

To view the Dashboard, ["click here."](https://public.tableau.com/app/profile/mathias.w.boissevain/viz/Hotel_Review_Capstone/HotelDashboard)

___
## Table of Contents

- [Customer Review Analysis](#customer-review-analysis)
  - [Table of Contents](#table-of-contents)
  - [Project Summary](#project-summary)
  - [Project Planning](#project-planning)
    - [Project Goals](#project-goals)
    - [Project Description](#project-description)
    - [Initial Questions](#initial-questions)
  - [Data Dictionary](#data-dictionary)
  - [Outline of Project Plan](#outline-of-project-plan)
    - [Data Acquisition](#data-acquisition)
    - [Data Preparation](#data-preparation)
    - [Exploratory Analysis](#exploratory-analysis)
    - [Modeling](#modeling)
    - [Deliverables](#deliverables)
  - [Lessons Learned](#lessons-learned)
  - [Instructions For Recreating This Project](#instructions-for-recreating-this-project)

___
## Project Summary

Our analysis of hotel customer reviews found the following:
  - Couples and leisure travelers were the groups who are generally most pleased with the hotels observed in this project. 
  - The most valuable insights about customer satisfaction are found in the positive reviews. Sentiment analysis and topic model clustering show that indicators from negative reviews are significantly more muted.
  
  - The areas of highest interest, as indicated by topic modeling, are:
    - location
    - staff
    - room
    - food

- We coalesced the results of our analysis into an interactive dashboard that provides individualized metrics for each hotel in the dataset.

___
## Project Planning

<details><summary><i>Click to expand</i></summary>

### Project Goals

The goal of this project it to provide actionable recommendations to our partner hotels on how to increase their performance ratings based on our analysis of their customer review data.

### Project Description

As the data science team at Booking.com, we analyzed the extensive customer review dataset for our partner hotels in the European region. Using natural language processing, sentiment analysis, and topic modeling we were able to identify key groups within the body of customers and key topic drivers of reviewers' scores. From the reviewers’ scores, we calculated a Net Promoter Score-styled metric to gauge customer satisfaction more accurately. The end product of these various analyses is an interactive dashboard that presents each hotel with a summary of its customer service performance.

### Initial Questions
- What words/topics are associated with positive or negative reviews?
- What are drivers of review score/average score?
- Which customer groups give the highest/lowest review scores?

 

</details>

___
## Data Dictionary

<details><summary><i>Click to expand</i></summary>

| Variable              | Meaning      |
|:-:| :-- |
|Hotel_Address| Address of hotel.|
|Review_Date| Date when reviewer posted the corresponding review.|
|Average_Score| Average Score of the hotel, calculated based on the latest comment in the last year.|
|Hotel_Name| Name of Hotel.|
|Reviewer_Nationality| Nationality of Reviewer.|
|Negative_Review| Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be 'No Negative'.|
|ReviewTotalNegativeWordCounts| Total number of words in the negative review.|
|Positive_Review| Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be 'No Positive'.|
|ReviewTotalPositiveWordCounts| Total number of words in the positive review.|
|Reviewer_Score| Score the reviewer has given to the hotel, based on his/her experience.|
|TotalNumberofReviewsReviewerHasGiven| Number of Reviews the reviewers has given in the past.|
|TotalNumberof_Reviews| Total number of valid reviews the hotel has.|
|Tags| Tags reviewer gave the hotel.|
|Days_Since_Review| Duration between the review date and scrape date.|
|Additional_Numberof_Scoring| This number indicates how many valid scores without review in there.|
|lat| Latitude of the hotel.|
|lng| longtitude of the hotel.|
|trip_type| Type of trip ('leisure', 'business', 'unknown').|
|nights_stayed| Number of nights stayed.|
|group_type| Type of group ('couple', 'solo traveler', 'group', 'family with young children', 'family with older children', 'travelers with friends').|
|nps_group| NPS-style grouping of customer based on review score(below 7: 'detractor', 7-9: 'passive', above 9: 'promoter').|
|neg_sentiment_score| Sentiment Intensity score of negative review.|
|neg_lem_sentiment_score| Sentiment Intensity score of lemmatized negative review.|
|review_total_negative_word_counts| Word count of negative review.|
|negative_unique_word_count| Unique word count of negative review.|
|negative_topic|Topic model designators for negative reviews.|
|pos_sentiment_score| Sentiment Intensity score of positive review.|
|positive_unique_word_count| Word count of negative review.|
|pos_lem_sentiment_score| Sentiment Intensity score of lemmatized positive review.|
|positive_topic|Topic model designators for positive reviews.|
|negative_clean_review| Negative review, cleaned with NLP techniques.|
|negative_lemma| Lemmatized version of negative review.|
|positive_clean_review| Positive review, cleaned with NLP techniques.|
|positive_lemma| Lemmatized version of positive review.|

</details>

___
## Outline of Project Plan

The following outlines the process taken through the Data Science Pipeline to complete this project.

Plan &#8594; Acquire &#8594; Prepare &#8594; Explore &#8594; Model &#8594; Deliver

---
### Data Acquisition

<details><summary><i>Click to expand</i></summary>

The data was collected by utilizing kaggles API to acess and pull the data from kaggle. The dataset, containing 515,738 customer reviews and scores for 1493 luxury hotels across Europe, was found on kaggle (originally scraped from Booking.com). All data in the file is publicly available. See the data dictionary above.

</details>

### Data Preparation

<details><summary><i>Click to expand</i></summary>

This project required extensive data cleaning and wrangling, including:
- changing the column names to all lower case
- parsing the list of strings in the tags column into separate feature columns
- changing the data type of the timestamp column and engineering additional features containing portions of the overall time stamp
- verifying and updating review word counts
- parsing the address values and creating separate features for country, city, etc.
- dropping unneeded columns
- preparing the text data from NLP including basic clean, removing stopwords, and lemmatizing
- changing the order of the columns within the dataframe
- cache the wrangled data as a json to reduce processing time during exploration
    
</details>

### Exploratory Analysis

<details><summary><i>Click to expand</i></summary>

- Who is the customer?
  - The initial exploration of the dataset consisted of reviewing the distribution of customers across key groupings including trip type, group type and nights stayed as well as looking at reviewer score distributions and average hotel score distributions.
- Reviewer scores
- Net Promoter Score-style groups and and an accompanying promoter score metric
- NLP
  - Word frequency
  - Sentiment analysis
  - Topic modeling
- Drivers of score by customer group
  - Group type &#8594; 'Couple'
  - Trip type &#8594; 'Leisure'
  - Nights stayed $\leq$ 3
- Breakdown by hotel
  - Overview of aggregated hotel data
  - General recommendations
- Summary
- Recommendations
    
</details>

### Modeling

<details><summary><i>Click to expand</i></summary>

- For topic modeling, we started with sklearn's Latent Dirichlet Allocation (LDA) model. The LDA model takes each word in the document and assigns it to one of k topics. k is the number of topics set as a hyperparameter in the creation of the LDA model. The model then calculates the proportion of words in the document that are assigned to each topic or $p(topic|document)$. It then calculates the proportion of documents assigned to each topic because of the words in the document or $p(word|topic)$. 
- The findings of the model were not conclusive enough to issue recommendations by themselves, so we added our own analysis of word frequency and context to the keyword groupings extracted by the model to generate a list of topics for each type of review.
    

</details>
___

### Deliverables

<details><summary><i>Click to expand</i></summary>

- The sum of our various analyses of this dataset is collected within an interactive dashboard. The dashboard houses the metrics and statistics individually most relevant to the customer service performance of each of the nearly 1,500 hotels served by Booking.com are presented for each hotelier to review at their leisure. ["Click here."](https://public.tableau.com/app/profile/mathias.w.boissevain/viz/Hotel_Review_Capstone/HotelDashboard)

</details>


## Lessons Learned

<details><summary><i>Click to expand</i></summary>

- Sentiment Intensity Analysis showed that guests who were on leisure trips had the most positive sentiment, and that solo travelers and families with young children tended to have lower positive sentiment than other groups. Sentiment intensity in negative reviews was mostly neutral, while in positive reviews, sentiment intensity was much more identifiably positive.

  &#8594;The "So What?": On a high level, better conclusions regarding areas in which a given hotel is doing well can be drawn from the positive reviews than can be drawn on areas in which that same hotel is underperforming as reported in the negative reviews. 

- Our LDA model identified dominant topics associated with the reviews for each hotel. Mapping of topic cluster segregations for positive and negative reviews also mirrored patterns found in sentiment analysis: negative reviews did not produce discernible clusters while positive reviews produced clearly stratified clusters. However, the model's output was not enough to develop actionable recommendations on its own. It was necessary to apply an analysis of word frequency and context on top of the model's groupings of keywords to produce a topic model that could point to specific opportunities for improvement or areas of excellence in providing hotel customers the best experience. 

  &#8594;The "So What?": Topics identified by our model, which combined the output of the unsupervised LDA algorithm with our human analysis of word frequency and context, come together to provide a hotelier with immediate insight on where to focus improvement efforts.

- Analysis of reviewers' scores was aided by the assignment of NPS-style customer groups and a pursuant calculation of a performance metric that effectively reduced the inherent inflation of scoring, thus providing a more accurate gauge of customer satisfaction.

  &#8594;The "So What?": Better conclusions about customer satisfaction can be drawn from the NPS-style metric, and the groups allow for further aggregation and analysis.

**Next Steps:**
- SHAP sentiment analysis

- Different topic modeling algorithms:
  - Truncated SVD/Latent Semantic Analysis
  - Non-negative Matrix Factorization
- Different topic model hyperparameters, vectorizers (TF/IDF)

    

</details>

___
## Instructions For Recreating This Project

<details><summary><i>Click to expand</i></summary>

1. Clone this repository into your local machine using the following command:
    
```bash
git clone git@github.com:InPersonAnalysis/customer_review_capstone.git
```

2. Retrieve the Kaggle API Token.
- pip install kaggle
- Log-in to Kaggle (or sign up)
- Navigate to your Account page (click top-right profile picture)
- API section on the Kaggle Account page.
- Scroll down to the API section and click Create New API Token
- Save kaggle.json to (/Users/<username>/.kaggle/) or in the OSError message given when attempting to import kaggle.   
    
3. You will need Natural Language Tool Kit (NLKT), Pandas, Numpy, Matplotlib, Seaborn, and SKLearn installed on your machine.

4. Now you can start a Jupyter Notebook session and execute the code blocks in the `final_report.ipynb` notebook.


</details>

[Back to top](#customer-review-analysis)
