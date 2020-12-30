'''
Question number 2 : What is the advantage and disadvantage of L1 loss function(Least absolute deviation) and L2 loss function(Least squared errors)

My sample answer :

L2 loss function is preferred in most cases. But when the outliers in the dataset, L2 does not perform well. The reason behind is that L2 is affected by outliers because of the consideration of squared differences, it leads to much larger errors. 
L1 is not affected by outliers.
Summary :
Only take advantage of L2 when the dataset do not have outliers, and vice versa.  
'''
'''
Question number 3 :
'''
'''
File D5 : Normal way
'''
import numpy as np
import pandas as pd 
from numpy import genfromtxt
import matplotlib.pyplot as plt


path = ('/home/khoacao/Downloads/data.csv')

data = genfromtxt(path, delimiter = ',')

areas = data[:,0]
prices  = data[:,1]
data_size = areas.size
print("area",areas)
print("price",prices)
print("Data Size",data_size) 


# Showing the relationship between two columns

fig = plt.figure(figsize = (10,5))
plt.scatter(areas,prices)
plt.plot(areas,prices)
plt.xlabel("Areas")
plt.ylabel("Prices")
plt.title("Correlation")
plt.show()
plt.savefig('scatter.png')


#################### Setting formula ###########

def predict(x,w,b) : 
    return x*w + b # output z
# theta is a hyperparameter call function
def compute_loss(z,y,theta = 30) :
    if abs(z-y) <= theta :
        loss =   (z-y)*(z-y)
    else : # otherwise
        loss =   theta*abs(z-y) - 1/2*theta*theta
    return loss 
# gradient of loss squared error   
def gradient_squared_function(x,z,y):
    dw = 2*x*(y-z)
    db = 2*(z-y)
    return (dw,db)
# gradient of loss absolute error
def gradient_absolute_function(x,z,y,theta = 30): 
    dw = (theta*x*(z-y))/abs(z-y)
    db = (theta*(z-y))/abs(z-y) 
    return dw,db
#dw,db = gradient_absolute_function(z,y,theta = 30)
def selected_function_gradient(x,y,z,theta = 30) :
    if abs(z-y) <= theta :
        dw,db = gradient_squared_function(x,y,z)
    else :
        dw,db = gradient_absolute_function(x,y,z)

    return dw,db 
#dw,db = selected_function_gradient(x,y,z,theta = 30) 
# backpropagation and feedfordward and backforward
def updated_weights(n,dw,db,w,b):
    w_new = w - dw*n 
    b_new = b - db*n 
    return (w_new,b_new)
losses = []   
w = -0.34
b =  0.04
epoch_max = 10  
n = 0.01 # Learning_rate 
for epoch in range(epoch_max) :
    for i in range(data_size) :
        x = areas[i]
        y = prices[i] 
        print("X",x)
        print("Y",y)
        z = predict(x,w,b)
        print("Forecast",z)
        loss = compute_loss(z,y,theta = 30)
        losses.append(loss)
        print("Loss",loss)
        dw,db = selected_function_gradient(x,y,z,theta = 30)
        w_new,b_new = updated_weights(n,dw,db,w,b)
        print("New_weight",w_new)
        print("B new",b_new)
        

## Visualize loss

plt.plot(losses)
plt.show()
plt.savefig('losses_1.png')