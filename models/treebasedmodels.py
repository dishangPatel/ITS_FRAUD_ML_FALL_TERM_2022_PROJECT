import numpy as np
import pandas as pd
import gc
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn import metrics



def changeDType(df,flag=False):
    
    if(flag):
        numericDtype = ['int32','int64','float64','float32']
    
    for i in df.columns:
        if (df[i].dtype == 'int64' or df[i].dtype == 'int32'):
            df[i] = pd.to_numeric(df[i],downcast='integer')
        
        if (df[i].dtype == 'float64' or df[i].dtype == 'float32'):
            df[i] = pd.to_numeric(df[i],downcast='float')



# applied stratified k folding to counter overfitting

def stratifiedKFoldWithDecisionTree(df,params=None,splits=5):
    
    train,test = train_test_split(df,test_size=0.30,stratify= df['isFraud'])
    
    X = train.drop(['isFraud'],axis=1)
    Y = train['isFraud']
    
    X = X - X.min()/X.max()-X.min()
    
    if(params==None):
    	params = {'max_depth':10}
    
    
    model = DecisionTreeClassifier(**params)
    skf = StratifiedKFold(n_splits=splits)
    
    for train_index,test_index in skf.split(X,Y):
        
        X_train = X.iloc[train_index]
        Y_train = Y.iloc[train_index]
        
        X_test = X.iloc[test_index]
        Y_test = Y.iloc[test_index]
        
        model.fit(X_train,Y_train)
        
        print("Accuracy ", model.score(X_test,Y_test))
        
        prob = model.predict_proba(X_test)[:,1]
        
        auc = metrics.roc_auc_score(Y_test,prob)
        print("validation-AUC : ",auc,"\n===============================")
    
    
    X_test = test.drop(['isFraud'],axis=1)
    Y_test = test['isFraud']
    
    X_test = X_test - X_test.min()/X_test.max() - X_test.min()
    
    prob = model.predict_proba(X_test)[:,1]
        
    auc = metrics.roc_auc_score(Y_test,prob)
    print("Test - AUC : ",auc,"\n===============================")

    return model


def evaluateModel(df,model):
    
    df = df - df.min()/df.max()-df.min()
    
    y_pred = model.predict(df)
    
    result = pd.DataFrame(y_pred)
    
    result.columns = ['isFraud']
    result.to_csv("./result_treeBased.csv")
    
    print("Result file saved")
    


# load dataset 
dfTrain = pd.read_csv('./preprocessedTrainV3.csv') # provide correct input file name
dfTest = pd.read_csv('./preprocessedTestV3.csv')  # provide correct input file name


# downcasting the datatype

changeDType(dfTrain)
changeDType(dfTest)

print("Train Data Shape",dfTrain.shape)
print("Test Data Shape",dfTest.shape)


# Trial 1

# set hyper parameter whatever you want to use

# you can pass no. of splits for cross validation

params = {'max_depth':10}
model = stratifiedKFoldWithDecisionTree(dfTrain,params,7)
evaluateModel(dfTest,model)


