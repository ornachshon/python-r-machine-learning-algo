# polynomial regression 

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
'''from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)'''


# fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)


# visualising the linear regression results
plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg.predict(x), color = 'blue')
plt.title('truth or bluff (linear regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

# visualising the polynomial regression results
plt.scatter(x,y, color = 'red')
plt.plot(x,lin_reg_2.predict(x_poly),color = 'blue')
plt.title('truth or bluff (polynomial regressin)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()

