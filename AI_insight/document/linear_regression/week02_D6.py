import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
path = ('/home/khoacao/Downloads/data.csv')

data = genfromtxt(path, delimiter = ',')

areas = data[:,0]
prices  = data[:,1]
data_size = areas.size
print("area",areas)
print("price",prices)
print("Data Size",data_size) 

# plotting scatter
sns.set_theme(color_codes = True) 
fig = plt.figure(figsize = (10,5))
plt.scatter(areas,prices)
plt.plot(areas,prices) #
plt.xlabel("areas",color = 'red')
plt.ylabel("prices",color = 'blue')
plt.title("Visualize correlation")
plt.show()
plt.legend()
plt.savefig('correlation.png')

# Building linear regression with the implementation of vectorization
## Bulding data vector
data = np.c_[areas,np.ones((data_size,1))] # Bulding vector data with areas and bias = np.ones
def predict(theta,x) :
    return x.T.dot(theta) # theta*T*x 
def compute_loss(z,y,delta = 20):
    if abs(z-y) <= delta:
        loss = (z-y)*(z-y)
    else :
        loss = delta*abs(z-y) - 1/2*delta*delta
    return loss
def gradient_absolute_loss(x,z,y,delta):
    dtheta = (delta*x*(z-y))/abs(z-y)
    #db = (delta*(z-y))/abs(z-y) 
    return theta
def gradient_squared_loss(x,y,z) : 
    dtheta = 2*x*(y-z)
    return theta
def select_function(z,y,delta) :
    if abs(z-y) <= delta : 
       dtheta = gradient_absolute_loss(x,y,z,delta)
    else :
       dtheta = gradient_gradient_squared_loss(x,y,z)
    return dtheta
# update new parameters using hyperparameters
def updated_weights(dtheta,theta,n): 
    theta_new = theta - dtheta*n
    return theta_new
## Initialize paramters and hyperparameters
losses = [] # append to count sum and visualize 
thetas = []
n = 0.01 # Learning rate
theta = np.array([[-0.34],[0.04]]) # vector [w,b]
epoch_max = 1
m = 2 # mini batch_size
delta = 20
for epoch in range(epoch_max) :
    for i in range(0,data_size,m) : # loop each mini batch(m = 2)
        # performance measure x and y for linear regression
        for index in range(i,i+m) :
          print("index",index)
          x = data[index]
          y = prices[index]
          print("x",x)
          print("y",y)
          z = predict(theta,x)
          print("forecast_values",z)
          loss = compute_loss(z,y,delta = 30)
          print("loss",loss)
          losses.append(np.mean(loss)) # Visualize
          # updated_weights
          #dw,db = elect_function(z,y,delta) 
          #print("Update_New_weight",dw)
          #print("Update_New_Bias",db) 
          dtheta = select_function(z,y,delta)
          theta_new = updated_weights(dtheta,n,theta)
          print("New_weights",theta_new)
          thetas.append(theta_new)
          print("List of thetas",thetas)


## visualize mean loss

figure = plt.figure(figsize = (10,5))
plt.plot(losses)
plt.xlabel("iteration",color = 'blue')
plt.ylabel("losses in each iteration", color = 'blue')
plt.title("Losses observation") 
plt.show()
plt.savefig("observation.png")         

## visualize
'''
figure = plt.figure(figsize = (10,5))
plt.plot(thetas)
plt.xlabel("iteration",color = 'blue')
plt.ylabel("theta in each iteration", color = 'blue')
plt.title("theta observation") 
plt.show()
plt.savefig("thetas_update.png")         
'''