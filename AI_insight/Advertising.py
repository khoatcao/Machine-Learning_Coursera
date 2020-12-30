import pandas as pd   # pylint: d
import numpy as np      # pylint:
import matplotlib.pyplot as plt  # pylint:
from numpy import genfromtxt
import plotly.express as px


path = ('/home/khoacao/Downloads/advertising.csv')
'''
data = genfromtxt(path,delimiter=",", skip_header = 1)

print("Data size",data.size)
print(data)
'''
## Dataset
data = pd.read_csv(path)
print(data.head()) 
print("Columns name", data.columns)
print("Data null values",data.isnull().sum().sort_values())


## Features and Label
X = data.iloc[:,:3].values
y = data.iloc[:,3:].values
#print("Columns of independent variables",X.columns)
#print("Columns of dependent variables",y.columns) 
#### The popular way #####
# Splitting the dataset into training and testing set 80 for training and 20 for testing
n_training = 160
n_testing  = 40
# for X 
X_train = X[:n_training] # 160 
y_train  = y[:n_training] # 40
# for y
X_test = X[n_training:]
y_test = y[n_training:]
### Visualize Box plot to find outliers ###
fig = plt.figure(figsize = (10,7))
plt.boxplot(X,[2,3,4])
plt.show()
## From the figure. It can be clearly see that Outliers only be extracted from Newspapers column
#fig = px.box(X, y = [0,1,2])
#fig.show()
plt.savefig('Outliers.png')
# In this eposode we are not going to cover outliers topic, Thus we only focus on developing machine learning model to work with that
## In order to determine whether we go for standardization or normalization. Let me visualize the distribution of the data #####
fig = plt.figure(figsize = (5,5))
colors = ['green','blue','red']
columns_name = data.columns
plt.hist(X,density = True,histtype = 'bar', color = colors, label = columns_name)
plt.legend(prop = {'size' :10})
plt.title("Distribution")
plt.savefig("Histogram result")
## data normalization ###
import seaborn as sns      
sns.set_theme(color_codes = True)
fig = plt.figure(figsize = (10,5))
name = data.columns
sns.distplot(X[0],kde = True, hist = False, label = name, color = "y")
plt.title("Normal Distribution in TV")
plt.show()
plt.savefig('Normal Distribution in TV')

################################################################################################
sns.set_theme(color_codes = True)
fig = plt.figure(figsize = (10,5))
colors = ['green','blue','red']
name = data.columns
sns.distplot(X[1],kde = True, hist = False, label = name, color = "y")
plt.title("Normal Distribution of Radio")
plt.show()
plt.savefig('Normal Distribution Radio')

####################################################################
sns.set_theme(color_codes = True)
fig = plt.figure(figsize = (10,5))
name = data.columns
sns.distplot(X[2],kde = True, hist = False, label = name, color = "y")
plt.title("Normal Distribution in Newspaper")
plt.show()
plt.savefig('Normal Distribution in Newspaper')

#### Find the best correlation between features and Label #######
fig = plt.figure(figsize = (10,5))
ax = fig.gca(projection = '3d')
ax.scatter(X[:,0],y,label = 'y', s= 5)
ax.legend()
ax.view_init(45,0)
plt.show()
plt.savefig('scatter.png')
### Data normailization ### #################################################### 
## formula = value - min/max
max = np.max(X_train)
min = np.min(X_train)
X_train = (X_train - min)/(max - min)
print("X train transform",X_train)
############# Starting Building model Machine Learning and ready training ################################ 


## Apply vectorization to reduce the hypothesis function to a single number with theta transposed and X matrix
## create feature vector [b,w]

X_training_new = np.c_[np.ones((n_training,1)),X_train]
print(data)

epoch_max = 20 # interate through each epoch
n = 0.01 # learning rate
## init theta
losses = []
theta = np.random.randn(4,1) # from random sample from N(M,o^2)
theta_path = []
## the hypothesis function
def predict(X_training_new,theta) : 
    return X_training_new.dot(theta) ## required output 

output =  predict(X_training_new,theta)
## Computing gradient for loss function 
def loss_function(output, y_train) :
    loss = np.abs(output - y_train) #|hypothesis - label |  
    return loss 
loss =  loss_function(output,y_train) 
def Loss_gradient(output,y) : # Compute gradient descent of loss function 
    d_loss = 2*(output - y_train)/n_training # Compute loss on one sample
    return d_loss 
d_loss =  Loss_gradient(output,y_train) 
def gradients(X_training_new,d_loss) : 
    gradient = X_training_new.T.dot(d_loss)
    return gradient
gradient_new = gradients(X_training_new,d_loss) 
def weights_update(theta,theta_path) : 
    theta = theta - n*gradient_new
    theta_path.append(theta) 
    return theta 
theta_new = weights_update(theta, theta_path)
# Compute average loss 
def mean_loss(loss, n_trainings) : 
    mean = np.sum(loss)/n_trainings 

    return mean

compute_mean = mean_loss(loss,n_training)
#################################################### ###################################################

############################## Loop through each epoch for training #################################### 

for epoch in range(epoch_max) :
    for i in range(n_training) :  
        print("Sample data",X_training_new[i])
        print("Sample data y",y_train[i])
        #output = predict(X_1,theta)
        #print("Output",output) 
        #loss = loss_function(output,y_1)
        #print("Loss",loss)
        print("Output",output)
        print("Loss",losses.append(loss[0]))
        print("Loss_new",loss)
        print("dloss",d_loss)
   #     print("Update theta",theta_new)
        print("Mean",compute_mean)



## Visualize losses
fig = plt.figure(figsize = (10,5))
X_axis = list(range(100))
plt.plot(X_axis,loss[:100], color = 'r')
plt.title("loss in 100 samples")
plt.show()
plt.savefig("losses.png")

5) 
''''
Accuracy can not be loss function. In some cases, we are either get high or high loss in classification problem

6) 

