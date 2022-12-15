import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats


import matplotlib.pyplot as plt

##################################### visualisations##################################
def viz_stroke_percentage(df):    
    fig, ax = plt.subplots(figsize=(9, 6),facecolor='floralwhite')

    sizes = np.array([len(df[df.stroke == 0]), len(df[df.stroke ==1])])
    labels = ["No Stroke", "Stroke Ocurred"]
    explode = [0.25,0 ]
    colors = ['#bad6eb', '#2b7bba']

    # Capture each of the return elements.
    patches, texts, pcts = ax.pie(
        sizes, labels=labels, autopct='%.2f%%', explode=explode,colors=colors,
        wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
        textprops={'size': 16})
    # Style just the percent values.
    plt.setp(pcts, color='white', fontweight='bold')
    ax.set_title('Percent of patient who have suffer a stroke', fontsize=25,color='#0b559f')
    plt.tight_layout();

def viz_hypertension_vs_stroke(df):    
    fig, ax = plt.subplots(figsize=(9, 7))
    ax=sns.barplot(x="hypertension", y='stroke', data=df, palette='Blues',edgecolor='white',linewidth=3)
    mean_line = df.stroke.mean()
    plt.axhline(mean_line, label='Stroke Rate', color='#0b559f',linestyle='-.', linewidth=3)

    ax.set(facecolor='floralwhite')
    ax.set_xlabel('Hypertension', fontsize=16)
    ax.set_ylabel('Stroke Rate', fontsize=16)
    ax.set_title('Hypertension Population', fontsize=30,color='#0b559f')

    ax.set_xticklabels(['No','Yes'],fontsize=15)
    plt.text(-.1, .055, 'Average Stroke Rate of the Population', fontsize = 15,color='black',rotation=0)
    plt.show();

def viz_heart_hyper_stroke(train):  
    fig, ax = plt.subplots(figsize=(16, 7))

    plt.subplot(122)
    ax=sns.barplot(x="hypertension", y='stroke', data=train, palette='Blues',edgecolor='white',linewidth=3)
    mean_line = train.stroke.mean()
    plt.axhline(mean_line, label='Stroke Rate', color='#0b559f',linestyle='-.', linewidth=3)

    ax.set(facecolor='floralwhite')
    ax.set_xlabel('Hypertension', fontsize=16)
    ax.set_ylabel('Stroke Rate', fontsize=16)
    ax.set_title('Hypertension Population', fontsize=30,color='#0b559f')
    ax.set_ylim(0, 0.200)
    ax.set_xticklabels(['No','Yes'],fontsize=15)
    plt.text(-.1, .055, 'Average Stroke Rate of the Population', fontsize = 15,color='black',rotation=0)

    plt.subplot(121)
    ax2=sns.barplot(x="heart_disease", y='stroke', data=train, palette='Blues',edgecolor='white',linewidth=3)
    mean_line = train.stroke.mean()
    plt.axhline(mean_line, label='Stroke Rate', color='#0b559f',linestyle='-.', linewidth=3)

    ax2.set(facecolor='floralwhite')
    ax2.set_xlabel('Heart disease', fontsize=16)
    ax2.set_ylabel('Stroke Rate', fontsize=16)
    ax2.set_ylim(0, 0.200)
    ax2.set_title('Heart Population', fontsize=30,color='#0b559f')

    ax2.set_xticklabels(['No','Yes'],fontsize=15)
    plt.text(-.1, .055, 'Average Stroke Rate of the Population', fontsize = 15,color='black',rotation=0);


def viz_gender_heart_stroke(train):    
    plt.figure(figsize=(9,7))
    ax = sns.barplot(data=train, x='gender_Male', y='stroke',hue='heart_disease',palette='Blues',
                     edgecolor='white',linewidth=3)
    plt.text(0, .055, 'Average Stroke Rate of the Population', fontsize = 15,color='black',rotation=0)
    mean_line = train.stroke.mean()
    plt.axhline(mean_line, label='Stroke Rate', color='#0b559f',linestyle='-.', linewidth=3)

    ax.set(facecolor='floralwhite')
    ax.set_xlabel('Gender', fontsize=16)
    ax.set_ylabel('Stroke Rate', fontsize=16)
    ax.set_title('Heart Disease by gender', fontsize=30,color='#0b559f')
    ax.set_xticklabels(['Female','Male'],fontsize=15);

def viz_age_vs_stroke(train):  
    plt.figure(figsize=(15,5))

    plt.subplot(121)
    ax= sns.histplot(x='age', data=train, hue='stroke',multiple='dodge', kde= True, bins = 8,)
    ax.set(facecolor='floralwhite')
    ax.set_xlabel('Age', fontsize=16)
    ax.set_ylabel('Population Count', fontsize=16)
    ax.set_title('Population vs Age', fontsize=30,color='#0b559f')

    plt.subplot(122)
    ax2 = sns.stripplot(data=train,y='age',x='stroke', alpha=0.5,jitter=True )
    ax2.set(facecolor='floralwhite')
    ax2.set_xlabel('Stroke ', fontsize=16)
    ax2.set_ylabel('Age', fontsize=16)
    ax2.set_title('Age vs Stroke', fontsize=30,color='#0b559f')
    #ax.set_yticklabels([0,100,200,300,400,500],fontsize=15)
    ax2.set_xticklabels(['No','Yes'],fontsize=15);


def viz_marriage_vs_stroke(train):    
    plt.figure(figsize=(10,5))

    ax= sns.countplot(data=train, x='ever_married_Yes', hue='stroke',palette='Blues')
    ax.set(facecolor='floralwhite')
    ax.set_xlabel('Ever Married', fontsize=16)

    ax.set_ylabel('Population Count', fontsize=16)
    ax.set_title('Marriage vs Stroke', fontsize=30,color='#0b559f')
    #ax.set_yticklabels([0,50,100,150,200,250],fontsize=15)
    ax.set_xticklabels(['No','Yes'],fontsize=15);
################################### Statistical test ####################################

def chi_square_test(df,target, feature):    
    '''
    chi_squre_test takes in a dataframe, target variable and feature 
    to create a crosstab between the target variable and feature and
    perform a chi square test
    returns 
    '''
    alpha = 0.05

    # Setup a crosstab of observed 
    observed = pd.crosstab(df[feature]== 1, df[target])

    t_stat, p_val, degf, expected = stats.chi2_contingency(observed)


    print(f' Chi-Square:{t_stat}')
    print(f' p-value:{p_val}')



def ttest(df,target, continuous_feature):
    ''' 
    ttest takes in a dataframe , categorical target and a continuous feature 
    to create two independent subgroubs base on categorical target and
    runs levene test to determine variance
    runs independent t-test
    returns t-stat and p-value
    '''
    # create two independent sample group of customers: churn and not churn.
    subset_feature =df[df[target]==1]
    subset_no_feature = df[df[target] == 0]

    # # stats Levene test - returns p value. small p-value means unequal variances
    stat, pval =stats.levene( subset_feature[continuous_feature], subset_no_feature[continuous_feature])
 

    # high p-value suggests that the populations have equal variances
    if pval < 0.05:
        variance = False

    else:
        variance = True

    
    # perform t-test
    t_stat, p_val = stats.ttest_ind(subset_feature[continuous_feature], subset_no_feature[continuous_feature],
                                    equal_var=variance,random_state=123)

    # round  and print results
    #t_stat = t_stat.round(4)
    #p_val = (p_val.round(4))/2
    print(f't-stat {t_stat}')
    print(f'p-value {p_val/2}')