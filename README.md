# Customer Review Analysis

This repository contains all files, and ipython notebooks, used in this project. A full outline of all the files with descriptions can be found below.

To view the Slide Deck, ["click here."](https://www.canva.com/design/DAFC1pAZhjY/37-GQyhTmSVtm3tACkxOEw/edit?utm_content=DAFC1pAZhjY&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 

To view the Handout, ["click here."](https://www.canva.com/design/DAFCXzznwGM/qiVjEIAp4Bx0uVw9UfR_Tg/edit?utm_content=DAFCXzznwGM&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 

To view the white paper, ["click here.](https://docs.google.com/document/d/1ph_axcpSNiGZAFXA7lmivC9EDDmzmvPa6Xr4bUssz6I/edit?usp=sharing)

To view the Final Report, ["click here."](https://github.com/InPersonAnalysis/customer_review_capstone/blob/main/final_notebook.ipynb)

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


___
## Project Planning

<details><summary><i>Click to expand</i></summary>

### Project Goals

The goal of this project it to provide actionable recommendations to our partner hotels on how to increase their ratings based on our analysis of their customer review data.

### Project Description

As the data science team at Booking.com, we analyzed the extensive customer review dataset for our partner hotels in the European region. Using natural language processing, sentiment analysis, and topic modeling we were able to identify key groups within the body of customers and key topic drivers of reviewers' scores. With the reviewers’ scores, we calculated a Net Promoter Score-styled metric for each hotel helping them understand their customer’s opinions so they can implement improvements based on the insight from our review analysis. 

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
|nps_group| NPS-style grouping of customer based on review score(below 6: 'detractor', 6-9: 'passive', above 9: 'promoter').|
|neg_sentiment_score| Sentiment Intensity score of negative review.|
|neg_lem_sentiment_score| Sentiment Intensity score of lemmatized negative review.|
|review_total_negative_word_counts| Word count of negative review.|
|negative_unique_word_count| Unique word count of negative review.|
|pos_sentiment_score| Sentiment Intensity score of positive review.|
|positive_unique_word_count| Word count of negative review.|
|pos_lem_sentiment_score| Sentiment Intensity score of lemmatized positive review.|
|negative_clean_review| Negative review, cleaned with NLP techniques.|
|negative_lemma| Lemmatized version of negative review.|
|positive_clean_review| Positive review, cleaned with NLP techniques.|
|positive_lemma| Lemmatized version of negative review.|

</details>

___
## Outline of Project Plan

The following outlines the process taken through the Data Science Pipeline to complete this project.

Plan &#8594; Acquire &#8594; Prepare &#8594; Explore &#8594; Model &#8594; Deliver

---
### Data Acquisition

<details><summary><i>Click to expand</i></summary>

The dataset, holding 515,738 customer reviews and scoring of 1493 luxury hotels across Europe, was found on kaggle (originally scraped from Booking.com). All data in the file is publicly available. A data dictionary can be found above.

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
- Reviewer scores
- Net Promoter Score-style groups and metric
- NLP
  - Word frequency
  - Sentiment analysis
  - Topic modeling
- Drivers of score by customer group
  - Group type &#8594; 'Couple'
  - Trip type &#8594; 'Leisure'
  - Nights stayed $\leq$ 4
- Breakdown by hotel
  - Overview of aggregated hotel data
  - General recommendations
- Summary
- Recommendations
    
</details>

### Modeling

<details><summary><i>Click to expand</i></summary>

A Latent Dirichlet Allocation (LDA) model fed by our NLP efforts, was employed to determine the top words associated with a particular hotel. Predictive modeling is not a focus of this project.
    

</details>
___

## Lessons Learned

<details><summary><i>Click to expand</i></summary>

- Sentiment Intensity Analysis found that guests who were on leisure trips had the most positive sentiment, and that solo travelers and families with young children tended to have lower positive sentiment than other groups. Sentiment intensity in negative reviews was mostly neutral, while in positive reviews, sentiment intensity was much more identifiably positive.
  -The "So What?": On a high level, better conclusions regarding areas in which a given hotel is doing well can be drawn from the positive reviews than those that can be drawn on areas in which that same hotel is underperforming as reported in the negative reviews. Given that these are luxury hotels ($$$), the flatness of sentiment scores on negative reviews could be attributed to a customer's tendency to want to reinforce their belief that they made a good purchase.

- Our LDA model identified dominant topics associated with the reviews for each hotel. Mapping of topic cluster segregations for positive and negative reviews also mirrored patterns found in sentiment analysis: negative reviews did not produce discernible clusters while positive reviews produced clearly stratified clusters. We found that the majority of positive reviews focus on (Topic 0) while the most negative topic mentioned in the reviews is (Topic -A). 
  -The "So What?": Topics identified by our LDA model can provide a hotelier with immediate insight on where to focus improvement efforts.


**Next Steps:**
- SHAP sentiment analysis
- Different topic modeling algorithms:
  - Latent Semantic Analysis
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
