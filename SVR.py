# SVR



# importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values




#splitting the dataset into the traning set and test set
"""from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)"""


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


# fitting SVR  to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

#create your reressor here



#predicting a new result with polynomial regression
y_pred = regressor.predict(6.5)

# visualising the SVR results 
plt.scatter(x,y, color = 'red')
plt.plot(x ,regressor.predict(x), color = 'blue')
plt.title('truth or bluff (SVR)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()






