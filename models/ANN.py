import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics


def evaluateModel(df,model):
    
    df = df - df.min()/df.max()-df.min()
    
    y_pred = model.predict(df)
    
    result = pd.DataFrame(y_pred)
    
    result.columns = ['isFraud']
    result.to_csv("./result_xgb_final.csv")
    
    print("Result file saved")


def create_model():
	
	model = Sequential()

	model.add(Dense(input_dim=XTrain.shape[1],activation='relu',units = 64))
	model.add(Dense(units = 64,activation='relu'))
	model.add(Dense(units = 32,activation='relu'))
	model.add(Dense(units = 16,activation='relu'))
	model.add(Dense(units = 8,activation='relu'))
	model.add(Dense(units = 4,activation='relu'))
	model.add(Dense(units = 2,activation='relu'))
	model.add(Dense(units = 1,activation='sigmoid'))
	
	model.compile(metrics=[tf.keras.metrics.AUC()],loss='binary_crossentropy',optimizer='adam')

	model.summary()
	
	return model

dfTest = pd.read_csv('./preprocessedTestV3.csv')
dfTrain = pd.read_csv('./preprocessedTrainV3.csv')


train,test = train_test_split(dfTrain,test_size=0.25,stratify=dfTrain['isFraud'])

XTrain = train.drop(['isFraud'],axis=1)
YTrain = train['isFraud']

XTest = test.drop(['isFraud'],axis=1)
YTest = test['isFraud']

XTrain = XTrain - XTrain.min()/XTrain.max()-XTrain.min()
XTest = XTest - XTest.min()/XTest.max()-XTest.min()



model = create_model()

model.fit(XTrain,YTrain,epochs=10)

evaluateModel(XTest,model)
