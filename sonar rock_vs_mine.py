#######................Predicting whether a object with given specification wnder water is Rock or a Mine using Machine Learining................#######
import numpy as np                                                  # for using array
import pandas as pd                                                 # to be used for dataframes
from sklearn.model_selection import train_test_split                # for using function train_test split
from sklearn.linear_model import LogisticRegression                 # for using logistic regression model
from sklearn.metrics import accuracy_score                          # for finding accuracy of our model

#Data collection and Data processing

#loading the data set to pandas dataframe
sonar_data=pd.read_csv("Copy of sonar data.csv",header=None)
# print(sonar_data)
print(sonar_data.head())

#number of rows and columns
print("Shape is: ")
print(sonar_data.shape)

# sonar_data.describe()                                             #describe() gives statical measures of the data
print("Last column is: ")
print(sonar_data[60])

print(sonar_data[60].value_counts())                                #M-->MINE   R-->ROCK

print(sonar_data.groupby(60).mean())

#Seperating Data and Labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]
print("X is :")
print(X)
print("Y is :")
print(Y)

#Training and Test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.1,stratify=Y,random_state=1)        
                                                                    #test_size=0.1 means tests data is 10% of the total data
print("Shape of X,X_train and X_test: ")
print(X.shape,X_train.shape,X_test.shape)
# print(X_test)


#Model training -->LogisticRegression()
model=LogisticRegression()

#training the logistic regression model with training data
model.fit(X_train,Y_train)
LogisticRegression(C=1.0,class_weight=None,dual=False,fit_intercept=True,intercept_scaling=1,max_iter=100,multi_class='auto',n_jobs=None,penalty='l2', random_state=None,solver='lbfgs'
,tol=0.0001, verbose=0,warm_start=False)

#Model eveluation

#Accuracy on training data
                                                                    #any accuracy greater than 70% is good

X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)

print("Accuracy on training data : ",training_data_accuracy)        #0.8342245989304813


#Accuracy on test data
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)

print("Accuracy on test data: ",testing_data_accuracy)              #0.7619047619047619

#Making a predictive system

input_data=(0.0307,0.0523,0.0653,0.0521,0.0611,0.0577,0.0665,0.0664,0.146,0.2792,0.3877,0.4992,0.4981,0.4972,0.5607,0.7339,0.823,0.9173,0.9975,0.9911,0.824,0.6498,0.598,0.4862,0.315,
0.1543,0.0989,0.0284,0.1008,0.2636,0.2694,0.293,0.2925,0.3998,0.366,0.3172,0.4609,0.4374,0.182,0.3376,0.6202,0.4448,0.1863,
0.142,0.0589,0.0576,0.0672,0.0269,0.0245,0.019,0.0063,0.0321,0.0189,0.0137,0.0277,0.0152,0.0052,0.0121,0.0124,0.0055)
                     # An example input data from the file data

#changing the input_data to numpy array
input_data_as_numpy_array=np.asarray(input_data)

#reshape the numpy array as we are prdicting for one instant
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)

print(prediction)
if prediction[0]=='R':
    print("The object is a Rock")
else:
    print("The object is a mine")
