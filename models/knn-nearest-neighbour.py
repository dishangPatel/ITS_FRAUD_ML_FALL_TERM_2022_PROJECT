import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os


# reducing memory for data by downcasting data type to proper size
def changeDType(df,flag=False):
    
    if(flag):
        numericDtype = ['int32','int64','float64','float32']
    
    for i in df.columns:
        if (df[i].dtype == 'int64' or df[i].dtype == 'int32'):
            df[i] = pd.to_numeric(df[i],downcast='integer')
        
        if (df[i].dtype == 'float64' or df[i].dtype == 'float32'):
            df[i] = pd.to_numeric(df[i],downcast='float')


train=pd.read_csv('./preprocessedTrainV3.csv') #train, provide correct input file name
test = pd.read_csv('./preprocessedTestV3.csv') #test, provide correct output file name

changeDType(train,False)
changeDType(test,False)


# making input ready for our model to be trained and tested
x=train.drop(['isFraud'],axis=1)
x = x - x.min()/x.max()-x.min()


#output for training and testing
y=train['isFraud']


#splitting our data into train-test
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=9) #Split the dataset


k = 100 # change different value of k for different model of different neighbour
model = KNeighborsClassifier(n_neighbors = k, n_jobs=-1).fit(X_train,y_train)
Pred_y = model.predict(X_test)
print("Accuracy of model on validation test data is",metrics.accuracy_score(y_test, Pred_y))
print("AUC Score of model on validation test data is",metrics.roc_auc_score(y_test, Pred_y))


# **Final Prediction**

#preparing test data for final prediction
test = test - test.min()/test.max()-test.min()

#final prediction
y_final=knn.predict(test)

#store into Dataframe
result = pd.DataFrame(y_final)

#saving as CSV file
result.to_csv('./result_knn_try.csv')


