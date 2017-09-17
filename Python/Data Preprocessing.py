# Data Preprocessing

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Data.csv')
np.set_printoptions(threshold = np.nan)
#creating a matrix of features [rows:columns (take all except the last one)]
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

#fixing missing data by filling in the mean of the column 
#use control+i to find the details about a class
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#fit the imputer to the matrix
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

#encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
#create dummy encoder class because there aint relation between the categorical values
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

#split the dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,t_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#do feature scaling (standardisation or normalisation)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)
