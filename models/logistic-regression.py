import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics



def changeDType(df,flag=False):
    
    if(flag):
        numericDtype = ['int32','int64','float64','float32']
    
    for i in df.columns:
        if (df[i].dtype == 'int64' or df[i].dtype == 'int32'):
            df[i] = pd.to_numeric(df[i],downcast='integer')
        
        if (df[i].dtype == 'float64' or df[i].dtype == 'float32'):
            df[i] = pd.to_numeric(df[i],downcast='float')


def evaluateModel(df,model):
    
    df = df - df.min()/df.max()-df.min()
    
    y_pred = model.predict(df)
    
    result = pd.DataFrame(y_pred)
    
    result.columns = ['isFraud']
    result.to_csv("./result.csv")
    
    print("Result file saved")
    

train = pd.read_csv('./preprocessedTrainV3.csv') # provide correct input file names
test=pd.read_csv('./preprocessedTestV3.csv')# provide correct input file names

changeDType(train,False)
changeDType(test,False)


# splitted the train data into 2 parts to do validation 
training_data, testing_data = train_test_split(train, test_size=0.3, random_state=50)


# creating Logistic Regression
model = LogisticRegression(solver='liblinear', random_state=0)


# preparing data for model fitting
x=training_data.drop(['isFraud'],axis=1)
y=training_data['isFraud']

# standardizing data points 
x=(x-x.mean())/(x.std())



# =================== Training Of Model ========================
# fit the model on data and print score
model.fit(x, y)
model.score(x, y)


#use model to predict probability that given y value is 1 or not

y_pred_proba = model.predict_proba(x)[::,1]

#calculate AUC of model
auc = metrics.roc_auc_score(y, y_pred_proba)

#print AUC score
print("AUC score on train data",auc)

# =================== Training Of Model End ========================



xtest=testing_data.drop(['isFraud'],axis=1)
ytest=testing_data['isFraud']
xtest=(xtest-xtest.mean())/(xtest.std())


y_pred = model.predict(xtest)

model.score(xtest, ytest)
y_pred_proba = model.predict_proba(xtest)[::,1]

#calculate AUC of model
auc = metrics.roc_auc_score(ytest, y_pred_proba)

#print AUC score
print("AUC score on validation test data",auc)

# ==================================== Hiddent Test Data =========================
evaluateModel(test,model)
