import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

######################################### Model Prep ###############################

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale,
               scaler=MinMaxScaler()):
    
    """
    Scales the 3 data splits. 
    Takes in train, validate, and test data 
    splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    
    Imports Needed:
    from sklearn.preprocessing import MinMaxScaler 
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import RobustScaler
    from sklearn.preprocessing import QuantileTransformer
    
    Arguments Taken:
               train = Assign the train DataFrame
            validate = Assign the validate DataFrame 
                test = Assign the test DataFrame
    columns_to_scale = Assign the Columns that you want to scale
              scaler = Assign the scaler to use MinMaxScaler(),
                                                StandardScaler(), 
                                                RobustScaler(), or 
                                                QuantileTransformer()
       return_scaler = False by default and will not return scaler data
                       True will return the scaler data before displaying the _scaled data
    """
    
    # make copies of our original data so we dont corrupt original split
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    # fit the scaled data
    scaler.fit(train[columns_to_scale])
    
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled

def model_prep(train,validate,test, modeling_features):
    
    ''' 
    model_prep takes in train, validate, test and features used for modeling,
    scales continuous variable
    removes features not used for modeling and 
    sepertes the target variable into its own dataframe
    returns x_train,y_train,x_validate,y_validate, x_test, y_test
        
    '''
    #scale continuous variables:
    scale_data(train, 
               validate, 
               test, 
               ['age'],
               scaler=MinMaxScaler())
    
    # drop unused columns and keep some features
    features = modeling_features
    train =train[features]
    validate = validate[features]
    test = test[features]
        
    #seperate target
    X_train = train.drop(columns=['stroke'])
    y_train = train.stroke

    X_validate = validate.drop(columns=['stroke'])
    y_validate = validate.stroke

    X_test = test.drop(columns=['stroke'])
    y_test = test.stroke
        
    # Convert binary categorical target variable to numeric
    return X_train,y_train,X_validate,y_validate, X_test, y_test

################################################ Models ###################################


def models_metrics(X_train,y_train, X_validate, y_validate): 
    '''
    models_metrics takes in X_train,y_train, X_validate, y_validate 
    makes 4 models : Decision tree, Random Forest, KNN neighbor, logistic regression
    fits models using train
    calculates accuracy and recall for train and validate data 
    returns metrics in a dataframe of models performance on train and validate data 
    '''  
    metrics= []
    # make tree model
    tree = DecisionTreeClassifier(max_depth=6, random_state=123)
    # Fit the model (on train and only train)
    tree = tree.fit(X_train, y_train)

    # accuracy on train
    in_sample_accuracy = tree.score(X_train, y_train)
    # accuracy on validate
    out_of_sample_accuracy = tree.score(X_validate, y_validate)

    # calculate recall train
    y_pred = tree.predict(X_train)
    in_sample_recall= recall_score(y_train, y_pred)  

    # calculate recall validate
    y_pred = tree.predict(X_validate)
    out_of_sample_recall= recall_score(y_validate, y_pred)

    output = {
        'Model': 'Decision Tree',
        "specs":'Max Depth = 6 ',
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy,
        'train_recall': in_sample_recall,
        'validate_recall': out_of_sample_recall
    }

    metrics.append(output)

    # Make the model
    random_forest = RandomForestClassifier(max_depth=8, min_samples_leaf = 1 , random_state=123)

    # Fit the model (on train and only train)
    random_forest = random_forest.fit(X_train, y_train)

    # accuracy on train
    in_sample_accuracy = random_forest.score(X_train, y_train)
    # accuracy on validate
    out_of_sample_accuracy = random_forest.score(X_validate, y_validate)

    # calculate recall train
    y_pred = random_forest.predict(X_train)
    in_sample_recall= recall_score(y_train, y_pred)  

    # calculate recall validate
    y_pred = random_forest.predict(X_validate)
    out_of_sample_recall= recall_score(y_validate, y_pred)


    output = {
        'Model': 'RandomForest',
        "specs":'Max Depth = 8,min_sample_leaf =1 ',
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy,
        'train_recall': in_sample_recall,
        'validate_recall': out_of_sample_recall
    }

    metrics.append(output)



    # create Knn model   
    knn = KNeighborsClassifier(n_neighbors=7, weights='uniform')
    # Knn fit model
    knn.fit(X_train, y_train)

    # accuracy on train
    in_sample_accuracy = knn.score(X_train, y_train)
    # accuracy on validate
    out_of_sample_accuracy = knn.score(X_validate, y_validate)

    # calculate recall train
    y_pred = knn.predict(X_train)
    in_sample_recall= recall_score(y_train, y_pred)  

    # calculate recall validate
    y_pred = knn.predict(X_validate)
    out_of_sample_recall= recall_score(y_validate, y_pred)



    output = {
        'Model': 'KNN neighbor',
        "specs":'neighbors =7, weights= uniform ',
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy,
        'train_recall': in_sample_recall,
        'validate_recall': out_of_sample_recall
    }

    metrics.append(output)

    # create the model    
    logit = LogisticRegression(C=1000, random_state=123, fit_intercept=True,solver ='saga' )
    # fit the model
    logit.fit(X_train, y_train)

    # accuracy on train
    in_sample_accuracy = logit.score(X_train, y_train)
    # accuracy on validate
    out_of_sample_accuracy = logit.score(X_validate, y_validate)

    # calculate recall train
    y_pred = logit.predict(X_train)
    in_sample_recall= recall_score(y_train, y_pred)  

    # calculate recall validate
    y_pred = logit.predict(X_validate)
    out_of_sample_recall= recall_score(y_validate, y_pred)


    output = {
        'Model': 'Logistic Regression',
        "specs":'solver = saga ',
        "train_accuracy": in_sample_accuracy,
        "validate_accuracy": out_of_sample_accuracy,
        'train_recall': in_sample_recall,
        'validate_recall': out_of_sample_recall
    }

    metrics.append(output)
    metrics_df = pd.DataFrame(metrics)
    return metrics_df