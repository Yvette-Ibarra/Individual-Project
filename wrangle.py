import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
################################### get data from csv file ####################################
def get_stroke_data():
    '''get_stroke_data takes retrives stroke data from csv file saved locally
    input:
        none
    output:
        df - data frame
    '''
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    return df

#################################### Prepare Data #############################################

def dummy(df, columns):
    '''
    dummy_var takes in data frame and column list and creates dummy variables for columns
    appends dummy to main data frame
    
    returns data frame
    '''
    # create dummies
    dummy_df = pd.get_dummies(df[
                                    columns], dummy_na=False)
    
    # Concatenate dummy_df to original data frame
    df = pd.concat([df, dummy_df], axis=1)
    
    return df


def rename_columns_and_values(df):
    ''' 
    rename_columns_and_values renames columns and values 
    returns dataframe
    input:
        df - dataframe
    output:
        df- dataframe
    '''
    # rename column to lowercase 
    df.rename(columns = {'Residence_type':'residence_type'}, inplace=True)
    
    # renames values of smoking status 
    df.smoking_status.replace({'never smoked': 'never_smoked', 'Unknown':'unknown',
                               'formerly smoked':'formely_smoked'},inplace=True)
    return df

def data_prep(df):
    '''
    data_prep takes in a dataframe and prepares for exploration: renames columms, drops unuseful columns
    input:
        df = data frame
    output:
        df = data frame
    '''
    # use rename_columns_and_values functin to rename columns
    df = rename_columns_and_values(df)
    

    
    # drop columns with unnecessary information
    df.drop(columns = {'id'},inplace=True)
 
    
    # categorical columns that will have dummy variables
    dummy_columns=['gender', 'ever_married','work_type', 'residence_type','smoking_status']
    
    # use dummy function to get dummy columns
    df = dummy(df, dummy_columns)
    
    return df

####################################### Data Split ###########################################

def impute_bmi_median(train, validate, test):
    '''
    Takes in train, validate, and test, and uses train to identify the best value to replace nulls in embark_town
    Imputes that value into all three sets and returns all three sets
    '''
    imputer = SimpleImputer(missing_values = np.nan, strategy='median')
    train[['bmi']] = imputer.fit_transform(train[['bmi']])
    validate[['bmi']] = imputer.transform(validate[['bmi']])
    test[['bmi']] = imputer.transform(test[['bmi']])
    return train, validate, test

def split_and_impute_data(df, target):
    '''
    split_date takes in a dataframe  and target variable and splits into train , validate, test 
    and stratifies on target variable
    
    The split is 20% test 80% train/validate. Then 30% of 80% validate and 70% of 80% train.
    Aproximately (train 56%, validate 24%, test 20%)
    
    returns train, validate, and test 
    '''
    # split test data from train/validate
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123, 
                                        stratify=df[target])

    # split train from validate
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123, 
                                   stratify=train_validate[target])
    # use impute_bmi_median function to fill in nulls in bmi columns after data split
    train, validate, test =impute_bmi_median(train, validate, test)
                                   
    return train, validate, test