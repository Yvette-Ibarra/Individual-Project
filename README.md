# Brain Stroke Classification Project

____
# Project Description:
According to the Center for Disease and Prevention (CDC) Every 40 seconds, someone in the United States has a stroke. Every 3.5 minutes, someone dies of stroke. This project uses  parameters like gender, age, various diseases, and smoking status to predict if a patient had a stroke. Each row in the data provides relavant information about the patient.
___
___
# Executive Summary:

#### Goals


To discover the drivers of stroke to predict to help our patients prevent a stroke occurance.
#### Key Findings
* 4.86% of the patient populatin has suffered a stroke
* Patients wih heart disease have a larger increase in stroke rate than patients with hypertension

* The patients age is a driver of stroke

#### Takeaways
* Top Model did not perform as expected, possible reason is data in imbalanced

#### Recommendation
In order to improve model more data collection and health information such as:

* if the patients  family has a history of stroke or heart disease
* Demographics of the patients 
* Patients weight and height
___
___

# Project Goal:
Discover drivers of stroke.
Use drivers to develop a machine learning model to predict weather or not a patient had a stroke.
Use findings to see what preventative  measures if any can be taken to prevent a stroke.
___
# Initial Thoughts:
My initial hypothesis is that hypertension driver of stroke.
___
# The Plan
* Acquire data from Kaggle

* Prepare data

* Explore data in search of drivers of stroke

* Answer the following initial questions:
    * 1. What is the percent of patients who have a stroke?
    * 2. Does the presense of hypertension increase the risk of stroke?
    * 3. Are patients with a heart condition more at risk of stroke than patients with hypertension?
    * 4. Controling for gender of a patient, does heart disease increases risk of stroke?
    * 5. Is age a driver of stroke?
    * 6. Do patients who have ever been married suffer more strokes than patients that have not been married?

* Use drivers identified in explore to build predictive models of different types
    * Evaluate models on train and validate data
    * Select the best model based on highest accuracy
    * Evaluate the best model on test data
    
* Draw conclusions
___
# Data Dictionary

|   Target Variable |Description|
| :------------- | -------------: | 
|       stroke    |  If the patient has a stroke or not. (1 = yes, 0 = no)     | 
___
___


 | Feature    | Description    | 
| :------------- | -------------: | 
|      id     |   unique identifier      | 
|   gender       |    Gender of patient Male or Female or Other     | 
| age|     age of the patient     | 
|      hypertension    |     If a patient has hypertension  (1 = yes, 0 = no)  | 
|      hear_disease     |     If a patient has any heart diseases  (1 = yes, 0 = no)  | 
|       ever_married   |   If a patient has ever been married (1 = yes, 0 = no)      | 
|    work_type       |    The work type of patient. *Children are under children category     | 
|  residence_type        |     If a patient residence is  rural of urban   | 
|   avg_glucose_level        |    average glucose level in blood     | 
|     bmi     |   body mass index of patient      | 
|   smoking_status       |    smoking status of patient * Unknown represent the information was unavailable     | 
|          |         | 

# Steps to Reproduce
    1. Clone this repository
    2. Get Telco Churn data in a csv from Kaggle: 
        https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset?select=healthcare-dataset-stroke-data.csv

    3. Save file in cloned repository
    
    4. Run notebook
---
# Takeaways and Conclusions
* 4.86% of the patient population  has suffered a stroke
* Patients with hypertension have an increase in stroke rate
* Patients with heart disease have an increase in stroke rate
* Patients with heart disease have a larger increase in stroke rate than patients with hypertension
* The male gender of our patients that have heart disease have a higher stroke rate than the female gender
* The patients age is a driver of stroke
* Patients who have been married have a slight increase in stroke rate.


---
# Recommendations
In order to improve model more data collection and health information such as:

* if the patients  family has a history of stroke or heart disease
* Demographics of the patients 
* Patients weight and height
    
Current top model is not ready to perform.
