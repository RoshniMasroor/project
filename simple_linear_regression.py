# Simple Linear Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Importing the dataset
data = pd.read_csv(r'D:\ML\KE.csv')
data.iloc[:, [0,1]].head()                      
data.columns = ['Date','Billed Amount']
data['Date'] =  pd.to_datetime(data['Date'], format='%m/%y')
data.set_index('Date', inplace=True)


# Create a time series plot.
plt.figure(figsize = (15, 5))
plt.plot(data, label = "Billed Amount")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("KE Future Prediction")
plt.legend()
data['Billed Amount'].plot()# set the index as x-axis for itself
plt.show()


print(data.reset_index())
X = data.iloc[:,'start_Index' ].values
y = data.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("KE Future Prediction")
plt.legend()
data['Billed Amount'].plot()# set the index as x-axis for itself
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')

#Predicting for a new ID number
print ('The electricity bill in 2021 will be')
print (regressor.predict([[2021]]))

