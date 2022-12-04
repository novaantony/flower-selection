# python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
import joblib
from django.http import HttpResponse
from django.shortcuts import render


data=pd.read_csv(r'C:\Users\novaa\Desktop\model\flower_nursery_dataset2.csv')
data=data.fillna(0)
data
data['Month']=le.fit_transform(data['Month'])
data['Season_blooms_from']=le.fit_transform(data['Season_blooms_from'])
data['Flower_family']=le.fit_transform(data['Flower_family'])
data['Color']=le.fit_transform(data['Color'])
data['Shape']=le.fit_transform(data['Shape'])

x=data[['Season_blooms_from','Flower_family','Color','Shape','Height_feet_max','Min_price','Max_price']]
y=data.Flower_Name

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.15)
model=RandomForestClassifier(n_estimators=12)
model=model.fit(xtrain,ytrain)

print('accuracy is ',model.score(xtest,ytest))

filename='finalmodeldataset.sav'
joblib.dump(model,filename)



# Create your views here.
def home(request):
    return render(request,'home.html')

def result(request):
    # return render(request,'result.html')
    model=joblib.load('finalmodeldataset.sav')


    lis=[]
    
    lis.append(request.GET['Season_blooms_from'])
    lis.append(request.GET['Flower_family'])
    lis.append(request.GET['Color'])
    lis.append(request.GET['Shape'])
    lis.append(request.GET['Height_feet_max'])
    lis.append(request.GET['Min_price'])
    lis.append(request.GET['Max_price'])
 
    ans=model.predict([lis])
    return render(request,'result.html',{'ans':ans})
