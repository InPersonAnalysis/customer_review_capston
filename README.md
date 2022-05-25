# Customer Review Analysis

This repository contains all files, and ipython notebooks, used in this project. A full outline of all the files with descriptions can be found below.

To view the slide deck, ["click here."]() 


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



___

## Project Planning

<details><summary><i>Click to expand</i></summary>

### Project Goals



### Project Description



### Initial Questions



</details>

___

## Data Dictionary

<details><summary><i>Click to expand</i></summary>


| Variable              | Meaning      |
| --------------------- | ------------ |
|Hotel_Address| Address of hotel.|
|Review_Date| Date when reviewer posted the corresponding review.|
|Average_Score| Average Score of the hotel, calculated based on the latest comment in the last year.|
|Hotel_Name| Name of Hotel|
|Reviewer_Nationality| Nationality of Reviewer|
|Negative_Review| Negative Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be| 'No Negative'|
|ReviewTotalNegativeWordCounts| Total number of words in the negative review.|
|Positive_Review| Positive Review the reviewer gave to the hotel. If the reviewer does not give the negative review, then it should be| 'No Positive'|
|ReviewTotalPositiveWordCounts| Total number of words in the positive review.|
|Reviewer_Score| Score the reviewer has given to the hotel, based on his/her experience|
|TotalNumberofReviewsReviewerHasGiven| Number of Reviews the reviewers has given in the past.|
|TotalNumberof_Reviews| Total number of valid reviews the hotel has.|
|Tags| Tags reviewer gave the hotel.|
|Days_Since_Review| Duration between the review date and scrape date.|
|Additional_Numberof_Scoring| This number indicates how many valid scores without review in there.|
|lat| Latitude of the hotel|
|lng| longtitude of the hotel|

</details>

___

## Outline of Project Plan

The following outlines the process taken through the Data Science Pipeline to complete this project.

Plan &#8594; Acquire &#8594; Prepare &#8594; Explore &#8594; Model &#8594; Deliver

---
### Data Acquisition

<details><summary><i>Click to expand</i></summary>

**Acquisition Files:**



**Steps Taken:**



</details>

### Data Preparation

<details><summary><i>Click to expand</i></summary>

**Preparation Files:**

- prepare.ipynb: Contains instructions for preparing the data and testing the prepare.py module.
- prepare.py: Contains functions used for preparing the readme's for exploration and modeling.
- preprocessing.py: Contains functions used for preprocessing data for exploration and modeling such as splitting data.

**Steps Taken:**



</details>

### Exploratory Analysis

<details><summary><i>Click to expand</i></summary>

**Exploratory Analysis Files:**

- explore.ipynb: Contains all steps taken and decisions made in the exploration phase with key takeaways.
- explore.py: Contains functions used for producing visualizations and conducting statistical tests in the final report notebook.

**Steps Taken:**

- First the data is split into three datasets: train, validate, and test. The training dataset is explored in the explore notebook and used later for training machine learning models. The validate and test datasets are used as unseen data to determine how the machine learning models perform on unseen data.


</details>

### Modeling

<details><summary><i>Click to expand</i></summary>

**Modeling Files:**

- model.ipynb: Contains all steps taken and decisions made in the modeling phase with key takeaways.
- model.py: Modeling procedures functionized for final report.

**Steps Taken:**



</details>

___

## Lessons Learned

<details><summary><i>Click to expand</i></summary>



**Next Steps:**


</details>

___

## Instructions For Recreating This Project

<details><summary><i>Click to expand</i></summary>

1. Clone this repository into your local machine using the following command:
```bash
git clone git@github.com:InPersonAnalysis/customer_review_capstone.git
```
2. You will need Natural Language Tool Kit (NLKT), Pandas, Numpy, Matplotlib, Seaborn, and SKLearn installed on your machine.

3. Now you can start a Jupyter Notebook session and execute the code blocks in the `final_report.ipynb` notebook.

</details>

[Back to top](#customer-review-analysis)
