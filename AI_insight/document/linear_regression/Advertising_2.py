import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from numpy import genfromtxt
import seaborn as sns
import plotly.express as px

path = ('/home/khoacao/Downloads/advertising.csv')
data = genfromtxt(path,delimiter = ',')
print(len(data))
data_1 = pd.DataFrame(data)
#print(data_1.iloc[1:,:])

data_1.rename(columns = {0:"TV",1:"Radio",2:"Newspaper",3:"Sales"},inplace = True)

print(data_1.head())

## plotting scatter 
sns.set_theme(color_codes = True)
figure = plt.figure(figsize = (10,5))
plt.scatter(data_1["TV"],data_1["Sales"])
#plt.plot(data_1["Sales"])
plt.xlabel("TV",color = 'red')
plt.ylabel("Sales",color = 'red')
plt.title("Correlation between TV and Sales")
plt.legend()
plt.show()
plt.savefig('Sale_Correlation_TV.png')

#### plotting scatter with Radio
sns.set_theme(color_codes = True)
figure = plt.figure(figsize = (10,5))
plt.scatter(data_1["Radio"],data_1["Sales"])
#plt.plot(data_1["Sales"])
plt.xlabel("Radio",color = 'red')
plt.ylabel("Sales",color = 'red')
plt.title("Correlation between Radio and Sales")
plt.legend()
plt.show()
#plt.savefig('Sale_Correlation_Radio.png')

#### with newspaper
sns.set_theme(color_codes = True)
figure = plt.figure(figsize = (10,5))
plt.scatter(data_1["Newspaper"],data_1["Sales"])
#plt.plot(data_1["Sales"])
plt.xlabel("Newspaper",color = 'red')
plt.ylabel("Sales",color = 'red')
plt.title("Correlation between TV and Sales")
plt.legend()
plt.show()
plt.savefig('Sale_Correlation_Newspaper.png')


## At the end I will go for TV is the best correlation with Sales
## get the independent variable and dependent variable
a = data_1.iloc[:,0].values
b = data_1.iloc[:,-1:].values
print(len(a))
print(len(b))
print("X",a)
print("y",b)
## splitting into training and testing set for 80 training and 20 n_testing
n_training = 160
n_testing = 40

X_train = a[:n_training] 
X_test = a[n_training:]

y_train = b[:n_training]
y_test = b[n_training:]
print("X_train",X_train)
print("Length of X_train",len(X_train))
print("Length of X_test",len(X_test))
print("Length of y_training",len(y_train))
print("Length of y_testing",len(y_test))
#print("print data X_train",X_train)
print("print data y_train",y_train)
print("Mean X_train",np.nanmean(X_train))

# draw box plot for X_train 
fig = plt.figure(figsize = (10,7))
sns.set_theme(color_codes = True)
sns.set_style("whitegrid")
sns.boxplot(x = "TV", data =  data_1)
plt.title("Outliers")
plt.xlabel("X_train")
plt.show()
plt.savefig("a.png")
# I did not see any outliers in boxplot
# in order to training model and get converence all we need to do is normalize data into the same range. Features will not dominant 

# Min - Max normalize and Standard Normalization
# In order to decide which feature scaling method to use. I firstly need to understand the distribution my data. 
# Min-Max normalization only uses in case the data is not a guassian distribution(bell curve)s
# standardization normalizaition only uses in case the data is a guassian distribution
import seaborn as sns      
sns.set_theme(color_codes = True)
fig = plt.figure(figsize = (10,5))
sns.distplot(data_1["TV"],kde = True, hist = False, label = "TV", color = "y")
plt.title("Normal Distribution in TV")
plt.show()
plt.savefig('Not_Gaussian.png')

# The Not_Gaussian figure shows that the dataset is not a gaussian Distribution so that I will use min max normalization. The objective of min max normalization
max = np.nanmax(X_train)
min = np.nanmin(X_train)
print("max",max)
print("min",min)

X_train = (X_train - min)/(max - min)
print("After normalization",X_train)

# Building model to display predict and compute gradient and update_weights
# linear function X*w + b = y
# using popular way
def predict(x,w,b) :
    return x*w + b

def compute_loss(z,y) :
    loss = (z-y)*(z-y) # using least squared error
    return loss
def compute_gradient(x,z,y) :
    dw = 2*x*(z-y)
    db = 2*(z-y)

    return (dw,db)

def update_weights(n,dw,db,w,b) :
    w_new = w - dw*n
    b_new = b - dw*n

    return (w_new,b_new)

# init parameters and hyperparameters

n = 0.01 # learning rate
w  = 0.04
b = -0.34
epoch_max = 1
m = 32 # batch_size(mini batch)
theta_w = []
theta_b = []
losses = []
for epoch in range(epoch_max) :
    for i in range(0,n_training,m) :
        for index in range(i,i+m) :
           x = X_train[index]
           y = y_train[index]
           print("X_index",x)
           print("y_index",y)
           z = predict(x,w,b)
           print("predict",z)
           loss = compute_loss(y,z)
           print("losses",loss)
           losses.append(loss)
           dw,db = compute_gradient(x,y,z)
           w_new,b_new = update_weights(n,dw,b,w,b)
           print("New weight",w_new)
           print("New bias",b_new)
           theta_w.append(w_new)
           theta_b.append(b_new)


# visualize losses
sns.set_theme(color_codes = True)
fig = plt.figure(figsize = (10,5))
plt.plot(losses)
plt.title("Losses Visualization")
plt.xlabel("iteration",color = 'red')
plt.ylabel("losses",color = 'blue')
plt.show()
plt.savefig("losses_2.png")

# update weights_update
sns.set_theme(color_codes = True)
fig = plt.figure(figsize = (10,5))
plt.plot(theta_w)
plt.title("update weights Visualization")
plt.xlabel("iteration",color = 'red')
plt.ylabel("weights",color = 'blue')
plt.show()
plt.savefig("weight_2.png")


##### update biases
sns.set_theme(color_codes = True)
fig = plt.figure(figsize = (10,5))
plt.plot(theta_b)
plt.title("update biases Visualization")
plt.xlabel("iteration",color = 'red')
plt.ylabel("biases",color = 'blue')
plt.show()
plt.savefig("biases_2.png")

#### Find the best parameters. It seems like hyperparameters tuning : losses = 3.7,b = 0.04 and w = 0.34
