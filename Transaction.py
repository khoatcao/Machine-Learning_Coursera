import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import re # regrex python 

path = ('/home/khoacao/Desktop/dataset/data1/ANZ_dataset.xlsx')

##### Exploratory data analysis###############################

def read_excel(path): 
    if True : 
        data = pd.read_excel(path)
        header = data.head()

        return header, data 
    else : 
        return None 
header, data = read_excel(path)
print(header)
print(data.head())
pd.set_option('display.max_columns',None)


def simple_data_analysis(data) : 
    # check columns 
    col = data.columns
    # check len rows 
    rows = len(data)
    # check missing values 
    Null_values = data.isnull().sum()
    # descriptive statistics 
    des = data.describe(include = np.number) 
    # cols type 
    cols_type = data.dtypes

    return  col, rows, Null_values, des, cols_type 

col, row, Null_values, des, cols_type = simple_data_analysis(data) 
print(col)
print(row)
print(Null_values)
print(des)
print(cols_type)

def cols_type(data): 
    
    for col in data.columns : 
        if data[col].dtype == "object" : 
            print("\n column name",col,)
            print(data[col].value_counts())
            print(data[col].isnull().sum())
        else : 
            print("\n columns name",col,)
            print("This is a numeric col",col)

print(cols_type(data))

 
## convert day, month,year seperately 
def format_date(data) :
    # date 
    date = data["date"]
    # covert to day, month,year seperately    
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year 

    return data['day'], data['month'], data['year'] 

print(format_date(data))
print(len(data.columns))



afashfasfhasfhasjfhasufhasfasfhsau fhaufhasfuhasfh
