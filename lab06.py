### It equals one for unsatisfied customers and 0 for satisfied customers 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
path_train = ('/home/khoacao/Desktop/dataset/lab06/train.csv')
path_test = ('/home/khoacao/Desktop/dataset/lab06/test.csv') 
df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test) 
print(df_train.head())
print(df_test.head())
# preprocessing for train and test 
# check columns and simple analysis

def basic_data_analysis(df_train,df_test) :
    # print length of dataset 
    print("The number of rows in the training set is ",len(df_train))
    print("The number of rows in the test set is ",len(df_test))
    print("The number of columns in the training set is ",len(df_train.columns))
    print("The number of columns in the test set is ",len(df_test.columns))
    # check number of columns
    cols_train = df_train.columns
    cols_test = df_test.columns  
    # statistical table for both train and test
    sta_train = df_train.describe(include  = np.number)
    sta_test  = df_test.describe(include = np.number) 
    # checking mssing values on both set with isnull method 
    mis_train = df_train.isnull().sum()
    mis_test = df_test.isnull().sum() 
    
    return cols_train,cols_test,sta_train,sta_test,mis_train,mis_test

cols_train,cols_test,sta_train,sta_test, mis_train,mis_test = basic_data_analysis(df_train,df_test)

print(cols_train)
print(cols_test)
print(sta_train)
print(sta_test)
print(mis_train)
print(mis_test)
print(pd.set_option('display.max_columns',None))

def target_variable(df_train) : 
    # target
    target = df_train.TARGET
    # value_counts
    val_count = target.value_counts(ascending = True)

    return val_count 

val_count = target_variable(df_train)

print(val_count) 
print("The total amount of unstatisfied customers nearly 30010")
print("The total amount of statisfied customer are 73012") 

def format_target(df_train) : 
    # label
    X = df_train.iloc[:,:-1]
    y = df_train.TARGET 
    
    return X,y 
X,y = format_target(df_train)
print(X)
print(y)

def format_data(X,y) : 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.15) 
    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test = format_data(X,y)
print(len(X_train.columns))
print(len(X_test.columns))
print(len(y_train))
print(len(y_test))
print("Write a function to split training set to another valid set ")
def format_valid(X_train,y_train) :
    X_train,X_valid,y_train,y_valid = train_test_split(X_train,y_train,test_size = 0.18, random_state = 43,shuffle = True)
    
    return X_train,X_valid,y_train,y_valid 

X_train,X_valid,y_train,y_valid = format_valid(X_train,y_train)

print(len(X_train))
print((X_valid.columns))
def normaliza_data(X_train,X_valid,StandardScaler) :
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train) 
    X_valid_scaled = std.transform(X_valid)
    return X_train_scaled,X_valid_scaled 

X_train_scaled,X_valid_scaled = normaliza_data(X_train,y_train,StandardScaler) 

print(X_train)
print(X_valid_scaled)


lfaj    skfajsfkajfkasjfsahfaksfhsakfhsaf
kashfsafhasfhsafjahsjasfjsafasfjsahfjsafhsajf