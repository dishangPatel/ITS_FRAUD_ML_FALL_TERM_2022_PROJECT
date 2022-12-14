import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



# To reduce RAM occupied by dataframe
def changeDType(df,flag=False):
    
    if(flag):
        numericDtype = ['int32','int64','float64','float32']
    
    for i in df.columns:
        if (df[i].dtype == 'int64' or df[i].dtype == 'int32'):
            df[i] = pd.to_numeric(df[i],downcast='integer')
        
        if (df[i].dtype == 'float64' or df[i].dtype == 'float32'):
            df[i] = pd.to_numeric(df[i],downcast='float')


# Evaluate pretrained model on test data
def evaluateModel(df,model):
    
    df = df - df.min()/df.max()-df.min()
    
    y_pred = model.predict(df)
    
    result = pd.DataFrame(y_pred)
    
    result.columns = ['isFraud']
    result.to_csv("./result.csv")
    
    print("Result file saved")


def naiveBayesGaussian(df,isTestData=False,model=None):
    
    
    train,test = train_test_split(df,stratify=df['isFraud'],test_size = 0.3,random_state = 9)
        
    nb = GaussianNB()
    
    x = train.drop(['isFraud'],axis = 1)
	y = train['isFraud']
    
    x = x - x.min()/x.max()-x.min()
    
    nb.fit(x,y)

# validation on test data
    xt = test.drop(['isFraud'],axis=1)
    yt = test['isFraud']

    print("Accuracy Score : " ,nb.score(xt,yt))
    prob = nb.predict_proba(xt)[:,1]

    auc = metrics.roc_auc_score(yt,prob)
    print("Validation AUC : ",auc)
    return nb



dfTest = pd.read_csv('./preprocessedTestV3.csv'); # provide name of the input files
dfTrain = pd.read_csv('./preprocessesTrainV3.csv');  # provide name of the input files


# Changing scaling down data type of feature whenever possible
changeDType(dfTest)
changeDType(dfTrain)


# Train the model on training data set
model = naiveBayesGaussian(dfTrain)


# Testing model on hidden data
evaluateModel(dfTest,model)


