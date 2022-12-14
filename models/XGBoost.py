import pandas as pd
import numpy as  np
from sklearn.model_selection import train_test_split,StratifiedKFold
from xgboost import XGBClassifier
from sklearn import metrics
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials,space_eval


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
    result.to_csv("./result_xgb_final.csv")
    
    print("Result file saved")
    
    
    
def xgbClassifier(df,params=None,splits=5):
    
    tmpPara = {'n_estimator':550,'eval_metric':'auc','verbosity':1,'n_jobs':-1,'reg_alpha':8.0,'colsample_bytree':0.8,'learning_rate':0.17500000000000002,'gamma':0.6000000000000001,'reg_lambda':1.20000000000000001,'max_depth':10,'min_child_weight':1.0,'scale_pos_weight':60.0,'subsample':0.700000000000001,'objective':'binary:logistic'}
    
    if(params!=None):
        tmpPara = params
    
    X_train = df.drop(['isFraud'],axis=1)
    Y_train = df['isFraud']
    
    X_train = X_train - X_train.min()/X_train.max() - X_train.min()
    
    skf = StratifiedKFold(n_splits=splits)
    xgbc = XGBClassifier(**tmpPara)
    
    for trainIndx,testIndx in skf.split(X_train,Y_train):
        
        Xtrain = X_train.iloc[trainIndx]
        Ytrain = Y_train.iloc[trainIndx]
        
        Xtest = X_train.iloc[testIndx]
        Ytest = Y_train.iloc[testIndx]
    
        xgbc.fit(Xtrain,Ytrain)
        
        print("Accuracy : ",xgbc.score(Xtest,Ytest))
        
        prob = xgbc.predict_proba(Xtest)[:,1]
        
        auc = metrics.roc_auc_score(Ytest,prob)
        print("validation-AUC : ",auc,"\n===============================")
        
        
    return xgbc


def objectiveFn(paramSpace):
    
    xgbc = XGBClassifier(**paramSpace)    
    xgbc.fit(XTrain,YTrain)
    
    prob = xgbc.predict_proba(XTest)[:,1]
    auc = metrics.roc_auc_score(YTest,prob)
    print("AUC : ",auc)
    return {'loss':-auc,'status':STATUS_OK} # -ve auc minimization ==> +auc maximization
    
def tuneParams():
    trialModels = Trials()
    
    bestParam = fmin(fn=objectiveFn,space=paramSpace,algo=tpe.suggest,max_evals=99,trials=trialModels)
    
    print(bestParam)
    return [bestParam,trialModels]



dfTest = pd.read_csv('./preprocessedTestV3.csv')
dfTrain = pd.read_csv('./preprocessedTrainV3.csv')


changeDType(dfTest)
changeDType(dfTrain)


train,test = train_test_split(dfTrain,test_size=0.25,stratify=dfTrain['isFraud'])

XTrain = train.drop(['isFraud'],axis=1)
YTrain = train['isFraud']

XTest = test.drop(['isFraud'],axis=1)
YTest = test['isFraud']

XTrain = XTrain - XTrain.min()/XTrain.max()-XTrain.min()
XTest = XTest - XTest.min()/XTest.max()-XTest.min()




paramSpace = {
    'n_estimators' : hp.randint('n_estimators',300,800),
    'max_depth' : hp.randint('max_depth',9,18),
    'learning_rate' : hp.quniform('learning_rate',0.01,0.2,0.015),
    'objective' : 'binary:logistic',
    'booster':'gbtree',
    'tree_method' : 'gpu_hist',
    'n_jobs': -1,
    'gamma' : hp.quniform('gamma',0.4,1,0.06),
    'min_child_weight' : hp.quniform('min_child_weight',1,12,1),
    'subsample':hp.quniform('subsample',0.55,1,0.055),
    'colsample_bytree':hp.quniform('colsample_bytree', 0.65, 1, 0.055),
    'reg_alpha' : hp.quniform('reg_alpha', 0, 10, 1),
    'reg_lambda': hp.quniform('reg_lambda', 1, 2, 0.12),
    'scale_pos_weight': hp.quniform('scale_pos_weight', 45, 200, 7), # helps in convergence for high imbalance
    'base_score' : hp.quniform('base_score',0.67,0.72,0.055),
    'eval_metric'  : 'auc',
}


# un comment this to tune the parameters....then use fine tuned parameters as best_params to predict 
# param_models = tuneParams()

# best params obtained from tunning
best_params = {'objective':'binary:logistic','n_jobs':-1,'base_score': 0.66, 'colsample_bytree': 0.935, 'gamma': 0.6, 'learning_rate': 0.03, 'max_depth': 15, 'min_child_weight': 10.0, 'n_estimators': 773, 'reg_alpha': 1.0, 'reg_lambda': 2.04, 'scale_pos_weight': 42.0, 'subsample': 0.935}

# it will train the model on training data
model = xgbClassifier(dfTrain,best_params)

# will evaluate trained model
evaluateModel(dfTest,model)





