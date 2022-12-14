import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,GridSearchCV,RandomizedSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def changeDType(df,flag=False):
    
    if(flag):
        numericDtype = ['int32','int64','float64','float32']
    
    for i in df.columns:
        if (df[i].dtype == 'int64' or df[i].dtype == 'int32'):
            df[i] = pd.to_numeric(df[i],downcast='integer')
        
        if (df[i].dtype == 'float64' or df[i].dtype == 'float32'):
            df[i] = pd.to_numeric(df[i],downcast='float')


dfTest = pd.read_csv('./preprocessedTestV3.csv')
dfTrain = pd.read_csv('./preprocessesTrainV3.csv')

# downcasting data type
changeDType(dfTrain)
changeDType(dfTest)


def stratifiedKFoldWithRandomForest(df):
    
    train,test = train_test_split(df,test_size=0.2)
    
    X = train.drop(['isFraud'],axis=1)
    Y = train['isFraud']
    
    X = X - X.min()/X.max() - X.min()
    
    model = RandomForestClassifier(n_estimators=100,n_jobs=-1)
    skf = StratifiedKFold(n_splits=5)
    
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
    
    prob = model.predict_proba(X_test)[:,1]
        
    auc = metrics.roc_auc_score(Y_test,prob)
    print("Test - AUC : ",auc,"\n===============================")

    return model


def evaluateModel(df,model):
    
    df = df - df.min() / df.max()-df.min()
    
    y_pred = model.predict(df)
    
    result = pd.DataFrame(y_pred)
    
    result.columns = ['isFraud']
    result.to_csv("./result_rf.csv")
    
    print("Result file saved")


# RandomSearchCV Algorithm to find best parameter for given data set 

def tuneParams(df,para=None):

	
	if(para==None):
		params = {'max_depth':10}
	else:
	    params = para
    
    rf = RandomForestClassifier()
    
    gridSearchCV = RandomizedSearchCV(estimator=rf,n_iter=30,cv=7,scoring='roc_auc',n_jobs=-1,verbose=2,param_distributions=params)
    
    train,test = train_test_split(df,test_size=0.23)
    
    X_train = train.drop(['isFraud'],axis=1)
    Y_train = train['isFraud']
    
    X_train = X_train - X_train.min()/X_train.max() - X_train.min()
    
    gridSearchCV.fit(X_train,Y_train)
    
    bestParams = gridSearchCV.best_params_
    
    print(bestParams)
    
    return gridSearchCV.best_estimator_



# Attemp No. 1 ==============================================================================================
model = stratifiedKFoldWithRandomForest(dfTrain)
evaluateModel(dfTest,model)



# Attempt no. 2 ==============================================================================================
# with fine tunned parameters
param = {
    'n_estimators' : [30,50,65,75,90,100,110,120,135,145,150,165,175,184,190,200,205,215],
    'criterion':['gini','entropy'],
    'max_depth' : [None,10,12,15,17,20,24,29,35,40,45,50],
    'min_samples_split' : [0.2,0.25,0.3,0.35,0.4,0.5,0.58,0.67,0.74,0.84,0.95,1.0],
    'max_features' : ['auto',0.2,0.3,0.4,0.5,0.6,0.65,0.7,0.75,0.8],
    'bootstrap' : [True],
    'max_samples' : [0.3,0.35,0.4,0.5,0.55,0.65,0.7,0.75,0.8],
    'n_jobs':[-1]
 }


model2 = tuneParams(dfTrain,param)
evaluateModel(dfTest,model2)



