# Brain Stroke Classification Project

# Project Description:
According to the Center for Disease and Prevention (CDC) Every 40 seconds, someone in the United States has a stroke. Every 3.5 minutes, someone dies of stroke. This project uses  parameters like gender, age, various diseases, and smoking status to predict if a patient had a stroke. Each row in the data provides relavant information about the patient.

# Project Goal:
Discover drivers of stroke.
Use drivers to develop a machine learning model to predict weather or not a patient had a stroke.
Use findings to see what preventitave measures if any can be taken to prevent a stroke.

# Initial Thoughts:
My initial hypothesis is that hypertension driver of stroke.

# The Plan
    * Aquire data from Kaggle

    * Prepare data

    * Explore data in search of drivers of stroke

    * Answer the following initial questions:
        *
        *
        *

    * Use drivers identified in explore to build predictive models of different types
        * Evaluate models on train and validate data
        * Select the best model based on highest accuracy
        * Evaluate the best model on test data
        
    * Draw conclusions

# Data Dictionary

|   Target Variable |Description|
| ------------- | -------------: | 
|       stroke    |  If the patient has a stroke or not. (1 = yes, 0 = no)     | 
|
|


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
# Takeaways and Conclusions

# Recommendations
