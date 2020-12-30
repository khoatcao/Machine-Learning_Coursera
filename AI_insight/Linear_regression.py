import numpy as np 
from numpy import genfromtxt 
import matplotlib.pyplot as plt 

path = ('/home/khoacao/Downloads/data.csv')

data = genfromtxt(path, delimiter = ',')

areas = data[:,0]
price  = data[:,1]
data_size = areas.size
print("area",areas)
print("price",price)
print("Data Size",data_size) 

# Data visualization

plt.scatter(areas,price, color = 'blue',label = "original data")
plt.plot(areas,price,color = 'red', label = 'fitting line')
plt.xlabel("areas")
plt.ylabel("price")
plt.title("The correlation")
plt.legend()
plt.show()
plt.savefig('figure.png')


#### forward ## 
def predict(x,theta) : 
    return x.dot(theta.T)

## Compute gradient ##
## derivative of Loss function 
def gradient(z,x,y) : # z is predicted,y is actual value, x is parameters
    dtheta = 2*x*(z-y)
    
    return dtheta

### update weight ###
def update_weight(theta,n,dtheta) :
    dtheta_new = theta - dtheta*n

    return dtheta_new 
data = np.c_[np.ones((data_size, 1)),areas]
print(data)
## init weight ### 
## learning rate ###
n = 0.01
theta = np.array([0.4,-0.34]) ## vector [b,w] 

# how long 
epoch_max = 10

## 
losses = [] # for debug

for epoch in range(epoch_max) :
    for i in range(data_size) : 
        # get sample
        x = data[i]
        y = price[i:i+1]
        ## compute predict 
        z = predict(x,theta)
        print("areas",x)
        print("price",y)
        print("forecast",z)
        ### compute loss

        loss = (z-y)*(z-y)
        losses.append(loss[0])
        print("loss",loss)
        # compute gradient
        dtheta = gradient(z,x,y) 
        print("dtheta",dtheta)
        ## update weight ####
        dtheta_new = update_weight(theta,n,dtheta) 

        print("dtheta_new",dtheta_new)

# visualize figure after updating parameters
'''
plt.scatter(x,z, label = 'forest data')
plt.plot(x,z,label = 'fitting line')
plt.xlabel("areas")
plt.ylabel("price")
plt.legend()
plt.show()
'''