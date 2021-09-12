from sklearn.metrics import confusion_matrix
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import TransformerMixin, BaseEstimator

import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

#Classes for data preprocessing
class CatFeaturesDummies(BaseEstimator, TransformerMixin):
    'The aim of this class is to select categorical features and to create dummy variables from these features'
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        _categorical = []
        categories = {
                      'Gender':['M','F'],'Education_Level':['High School', 'Graduate', 'Uneducated', 'Unknown', 'College',
                      'Post-Graduate', 'Doctorate'], 'Marital_Status':['Married', 'Single', 'Unknown', 'Divorced'], 'Income_Category':['$60K - $80K', 'Less than $40K', '$80K - $120K', '$40K - $60K',
                      '$120K +', 'Unknown'], 'Card_Category':['Blue', 'Gold', 'Silver', 'Platinum']
                        
                     }
        for col in X.columns :
            if X[col].dtypes == 'O' : _categorical.append(col)
                
        Xf = pd.DataFrame()
        
        for col in categories:
            Xf = pd.concat([Xf,pd.get_dummies(pd.Categorical(X[col], categories=categories[col]), drop_first=True, prefix=col)],axis=1)
 
        return Xf
    
class NumFeaturesSelection(BaseEstimator, TransformerMixin):
    'Selection of numerical features'
    
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self
    
    def transform(self,X):
        _numerical = []
        for col in X.columns :
            if X[col].dtypes in ['int64','float64'] : _numerical.append(col)
        Xt = X.copy()[_numerical]
        return Xt

class Scaler(BaseEstimator, TransformerMixin):
    'Transform the array obtained from scaling onto a DataFrame'
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        mms = MinMaxScaler()
        return pd.DataFrame(mms.fit_transform(X), columns=X.columns)

def preprocessDataAndPredict(

                                Customer_Age, Gender, Dependent_count,
                                Education_Level, Marital_Status, Income_Category, Card_Category,
                                Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon,
                                Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal,
                                Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt,
                                Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio,
                                Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1,
                                Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2
                            ):
    
    #keep all inputs in array
    col_names = ['Customer_Age', 'Gender', 'Dependent_count',
       'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category',
       'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
       'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
    test_data = [Customer_Age, Gender, Dependent_count,
                 Education_Level, Marital_Status, Income_Category, Card_Category,
                 Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon,
                 Contacts_Count_12_mon, Credit_Limit, Total_Revolving_Bal,
                 Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt,
                 Total_Trans_Ct, Total_Ct_Chng_Q4_Q1, Avg_Utilization_Ratio,
                 Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1,
                 Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2]
    print(test_data)
    
    #convert value data into numpy array
    test_data = pd.DataFrame(dict(zip(col_names, test_data)))
    
    #open file
    file = open("to_use_model.pkl","rb")
    
    #load trained model
    trained_model = joblib.load(file)
    
    #predict
    prediction = trained_model.predict(test_data)
    
    return prediction
    