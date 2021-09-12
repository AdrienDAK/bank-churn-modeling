#import Flask
from flask import Flask, render_template, request
from classes import*
import numpy as np
import joblib

#create an instance of Flask
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    
    if request.method == "POST":
        
        #get form data
        Customer_Age = request.form.get('Customer_Age')
        Gender = request.form.get('Gender')
        Dependent_count = request.form.get('Dependent_count')
        Education_Level = request.form.get('Education_Level')
        Marital_Status = request.form.get('Marital_Status')
        Income_Category = request.form.get('Income_Category')
        Card_Category = request.form.get('Card_Category')
        Months_on_book = request.form.get('Months_on_book')
        Total_Relationship_Count = request.form.get('Total_Relationship_Count')
        Months_Inactive_12_mon = request.form.get('Months_Inactive_12_mon')
        Contacts_Count_12_mon = request.form.get('Contacts_Count_12_mon')
        Credit_Limit = request.form.get('Credit_Limit')
        Total_Revolving_Bal = request.form.get('Total_Revolving_Bal')
        Avg_Open_To_Buy = request.form.get('Avg_Open_To_Buy')
        Total_Amt_Chng_Q4_Q1 = request.form.get('Total_Amt_Chng_Q4_Q1')
        Total_Trans_Amt = request.form.get('Total_Trans_Amt')
        Total_Trans_Ct = request.form.get('Total_Trans_Ct')
        Total_Ct_Chng_Q4_Q1 = request.form.get('Total_Ct_Chng_Q4_Q1')
        Avg_Utilization_Ratio = request.form.get('Avg_Utilization_Ratio')
        Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1 = request.form.get('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1')
        Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2 = request.form.get('Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2')
        
        #call preprocessDataAndPredict and pass inputs
        try:
            prediction = preprocessDataAndPredict(int(Customer_Age), str(Gender), int(Dependent_count),
                                Education_Level, Marital_Status, Income_Category, Card_Category,
                                int(Months_on_book), int(Total_Relationship_Count), int(Months_Inactive_12_mon),
                                int(Contacts_Count_12_mon), float(Credit_Limit), int(Total_Revolving_Bal),
                                float(Avg_Open_To_Buy), float(Total_Amt_Chng_Q4_Q1), int(Total_Trans_Amt),
                                int(Total_Trans_Ct), int(Total_Ct_Chng_Q4_Q1), float(Avg_Utilization_Ratio),
                                float(Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1),
                                float(Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2))


            #pass prediction to template
            return render_template('predict.html', prediction = prediction)
   
        except ValueError:
            return "Please Enter valid values"
  
        #pass
    pass

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
    test_data = [int(Customer_Age), Gender, int(Dependent_count),
                 Education_Level, Marital_Status, Income_Category, Card_Category,
                 int(Months_on_book), int(Total_Relationship_Count), int(Months_Inactive_12_mon),
                 int(Contacts_Count_12_mon), float(Credit_Limit), int(Total_Revolving_Bal),
                 float(Avg_Open_To_Buy), float(Total_Amt_Chng_Q4_Q1), int(Total_Trans_Amt),
                 int(Total_Trans_Ct), float(Total_Ct_Chng_Q4_Q1), float(Avg_Utilization_Ratio),
                 float(Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1),
                 float(Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2)]
    names = []
    for elt in test_data:
        names.append([elt])

    test_data = names
    #print(test_data)
    
    #convert value data into numpy array
    test_data = pd.DataFrame(dict(zip(col_names, test_data)))
    
    #open file
    file = open("to_use_model.pkl","rb")
    
    #load trained model
    trained_model = joblib.load(file)
    
    #predict
    prediction = trained_model.predict(test_data)
    
    return prediction


if __name__ == '__main__':
    app.run(debug=True)
