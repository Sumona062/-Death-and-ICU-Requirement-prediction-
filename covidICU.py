

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev
from sklearn.metrics import classification_report


#%%
df=pd.read_csv('covid.csv')
print(df.head())
#%%
print(df.isnull().sum())
#%%
covid= df.drop(['id','entry_date','date_symptoms','date_died'], axis = 1)

#%%
features = covid.columns
print(features)
#%%
feature= [x for x in features if x!= 'icu']
print(feature)
#%%
print(covid['covid_res'].value_counts().to_frame())
covid=covid[covid['covid_res']==1]
print(covid['covid_res'].value_counts().to_frame())
#%%
print(covid['icu'].value_counts().to_frame())
covid=covid[covid['icu']!=97]
covid=covid[covid['icu']!=99]
print(covid['icu'].value_counts().to_frame())
#%%
covid['age']=covid['age'].apply(lambda x:0 if x<18 else x)
covid['age']=covid['age'].apply(lambda x:1 if (x>=18 and x<=33) else x)
covid['age']=covid['age'].apply(lambda x:2 if (x>=34 and x<=49) else x)
covid['age']=covid['age'].apply(lambda x:3 if (x>=50 and x<=65) else x)
covid['age']=covid['age'].apply(lambda x:4 if x>65 else x)

print(covid['age'].value_counts().to_frame())

#%%
x=covid[feature]
y=covid['icu']
#%%
icu = ['icu needed','icu not needed']
#%%
dt= DecisionTreeClassifier(min_samples_split = 5000, criterion='entropy')
#%%
rf=RandomForestClassifier(n_estimators=400, criterion='entropy')   
#%%
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

accuracy_stratified = []

for train_index, test_index in skf.split(x,y):
    X_train_fold = x.iloc[train_index]
    X_test_fold = x.iloc[test_index]
    y_train_fold = y.iloc[train_index]
    y_test_fold = y.iloc[test_index]
    dt.fit(X_train_fold, y_train_fold)
    accuracy_stratified.append(accuracy_score(y_test_fold, dt.predict(X_test_fold)))
    
print("For Decision Tree:")
print('List of possible accuracy:', accuracy_stratified)
print('\nMaximum Accuracy:',
      round(max(accuracy_stratified)*100,4), '%')
print('\nMinimum Accuracy:',
      round(min(accuracy_stratified)*100,4), '%')
print('\nOverall Accuracy:',
      round(mean(accuracy_stratified)*100,4), '%')
print('\nStandard Deviation is:', stdev(accuracy_stratified))


#%%
train, test = train_test_split(covid, test_size = 0.20)
#%%
x_train = train[feature]
y_train = train['icu']

x_test = test[feature]
y_test = test['icu']

#%%
    
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test) 


#%% 
scoredt=accuracy_score(y_test, y_pred_dt)*100
print("Accuracy using Decision Tree:",round(scoredt, 2), "%")   
#%%
print(classification_report(y_test,y_pred_dt,target_names=icu))
#%%
plot_confusion_matrix(dt, x_test, y_test,cmap='PuBuGn',values_format='.1f',display_labels=icu)  
plt.title("Confusion matrix by decision tree")
plt.show() 

#%%
    
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test) 


#%% 
score=accuracy_score(y_test, y_pred_rf)*100
print("Accuracy using random forest:",round(score, 2), "%")   
#%%
print(classification_report(y_test,y_pred_rf,target_names=icu))


#%%
plot_confusion_matrix(rf, x_test, y_test,cmap='PuBuGn',values_format='.1f',display_labels=icu)  
plt.title("Confusion matrix by random forest")
plt.show()
#%%

tree.export_graphviz(dt, out_file="treeCicu.dot", feature_names=feature,  
                     class_names=icu,
                filled=True, rounded=True,
                special_characters=True,max_depth=3)

#%%

graph = pydotplus.graphviz.graph_from_dot_file("treeCicu.dot")
graph.write_png('treeCicu.png')
             