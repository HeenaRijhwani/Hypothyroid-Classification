#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.chdir(r"C:\Users\heena\Downloads")

df=pd.read_csv("hypothyroid.csv")


# In[58]:


df.head()


# In[59]:


df.tail()


# In[60]:


df.columns


# In[61]:


df.shape


# In[62]:


df.info()


# In[63]:


for column in df.columns:
    listOfValues=set(df[column])
    print(column,": ",listOfValues)


# In[64]:


df.dtypes


# In[65]:


df.describe().T


# In[66]:


df[["sex","T3","age","TBG","FTI","TT4","T4U","TSH","TBG measured"]]=df[["sex","T3","age","TBG","FTI","TT4","T4U","TSH","TBG measured"]].replace("?",np.nan)


# In[67]:


df[["age","TSH","T3","TT4","T4U","FTI"]]=df[["age","TSH","T3","TT4","T4U","FTI"]].astype(float)


# In[69]:


print(df.isnull().any())


# In[70]:


print(df.isnull().sum())


# In[71]:


#missingness map
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[72]:


df=df.drop("TBG",axis=1)
df = df.rename({'binaryClass':"Outcome"}, axis=1) 


# In[73]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df["Outcome"]=le.fit_transform(df["Outcome"].values)
df["sex"]=le.fit_transform(df["sex"].values)
df["on thyroxine"]=le.fit_transform(df["on thyroxine"].values)
df["query on thyroxine"]=le.fit_transform(df["query on thyroxine"].values)
df["on antithyroid medication"]=le.fit_transform(df["on antithyroid medication"].values)
df["sick"]=le.fit_transform(df["sick"].values)
df["pregnant"]=le.fit_transform(df["pregnant"].values)
df["thyroid surgery"]=le.fit_transform(df["thyroid surgery"].values)
df["I131 treatment"]=le.fit_transform(df["I131 treatment"].values)
df["query hypothyroid"]=le.fit_transform(df["query hypothyroid"].values)
df["query hyperthyroid"]=le.fit_transform(df["query hyperthyroid"].values)
df["lithium"]=le.fit_transform(df["lithium"].values)
df["goitre"]=le.fit_transform(df["goitre"].values)
df["tumor"]=le.fit_transform(df["tumor"].values)
df["hypopituitary"]=le.fit_transform(df["hypopituitary"].values)
df["psych"]=le.fit_transform(df["psych"].values)
df["TSH measured"]=le.fit_transform(df["TSH measured"].values)
df["T3 measured"]=le.fit_transform(df["T3 measured"].values)
df["TT4 measured"]=le.fit_transform(df["TT4 measured"].values)
df["T4U measured"]=le.fit_transform(df["T4U measured"].values)
df["FTI measured"]=le.fit_transform(df["FTI measured"].values)
df["TBG measured"]=le.fit_transform(df["TBG measured"].values)


# In[74]:


df0=df[df["Outcome"]==0] #healthy patients/ positive/ 1
df1=df[df["Outcome"]==1] #unhealthy patients/ negative/ 0


# In[75]:


df0["T3"].median()


# In[76]:


df0["TT4"].median()


# In[77]:


df0["T4U"].median()


# In[78]:


df0["FTI"].median()


# In[79]:


df1["T3"].median()


# In[80]:


df1["TT4"].median()


# In[81]:


df1["T4U"].median()


# In[82]:


df1["FTI"].median()


# In[83]:


df0["TSH"].median()


# In[84]:


df1["TSH"].median()


# In[85]:


df0["age"].median()


# In[86]:


df1["age"].median()


# In[87]:


def impute_T3(cols):
    T3=cols[0]
    Outcome=cols[1]
    if pd.isnull(T3): #if missing values found
        if Outcome == 0: #and if healthy patient
           return  1.5  #replace with 1.5
        else:
            return 2.0   #or else replace with 2.0
    else:
        return T3
df['T3']=df[['T3','Outcome']].apply(impute_T3, axis=1)  #applying the function


# In[88]:


def impute_TT4(cols):
    TT4=cols[0]
    Outcome=cols[1]
    if pd.isnull(TT4): #if missing values found
        if Outcome == 0: #and if healthy patient
           return  76.5 
        else:
            return 105.0   
    else:
        return TT4
df['TT4']=df[['TT4','Outcome']].apply(impute_TT4, axis=1)  #applying the function


# In[89]:


def impute_T4U(cols):
    T4U=cols[0]
    Outcome=cols[1]
    if pd.isnull(T4U): #if missing values found
        if Outcome == 0: #and if healthy patient
            return  1.01 
        else:
            return 0.97  
    else:
        return T4U
df['T4U']=df[['T4U','Outcome']].apply(impute_T4U, axis=1)  #applying the function


# In[90]:


def impute_FTI(cols):
    FTI=cols[0]
    Outcome=cols[1]
    if pd.isnull(FTI): #if missing values found
        if Outcome == 0: #and if healthy patient
            return 77.5  
        else:
            return 108.0  
    else:
        return FTI
df['FTI']=df[['FTI','Outcome']].apply(impute_FTI, axis=1)  #applying the function


# In[91]:


def impute_TSH(cols):
    TSH=cols[0]
    Outcome=cols[1]
    if pd.isnull(TSH): #if missing values found
        if Outcome == 0: #and if healthy patient
            return 12
        else:
            return 1.2 
    else:
        return TSH
df['TSH']=df[['TSH','Outcome']].apply(impute_TSH, axis=1)  #applying the function


# In[92]:


def impute_age(cols):
    age=cols[0]
    Outcome=cols[1]
    if pd.isnull(age): #if missing values found
        if Outcome == 0: #and if healthy patient
            return 55  
        else:
            return 54  
    else:
        return age
df['age']=df[['age','Outcome']].apply(impute_age, axis=1)  #applying the function


# In[93]:


df0['sex'].mode()
def impute_sex(cols):
    sex=cols[0]
    Outcome=cols[1]
    if pd.isnull(cols): #if missing values found
        if Outcome == 0: #and if healthy patient
            return 0
        else:
            return 1
    else:
        return sex


df['sex'].fillna(df['sex'].mode()[0], inplace=True)


# In[94]:


outliers =[] 
def detect_outliers(column):
    column1=sorted(column)  #sorting the data
    Q1,Q3 = np.percentile(column1 , [25,75]) #finding 1st and 3rd quartile 
    IQR = Q3-Q1  
    lower_range = Q1-(1.5 * IQR)
    upper_range = Q3+(1.5 * IQR)
    for x in column1: #for each record in specified column
        if ((x> upper_range) or (x<lower_range)):
             outliers.append(x)   #appending to list
    print(outliers)   
outlier=detect_outliers(df['T3'])


# In[95]:


outliers2 =[] 
def detect_outliers2(column2):
    column3=sorted(column2)  #sorting the data
    Q1,Q3 = np.percentile(column3 , [25,75]) #finding 1st and 3rd quartile 
    IQR = Q3-Q1  
    lower_range = Q1-(1.5 * IQR)
    upper_range = Q3+(1.5 * IQR)
    for x in column3: #for each record in specified column
        if ((x> upper_range) or (x<lower_range)):
             outliers2.append(x)   #appending to list
    print(outliers2)   
outlier2=detect_outliers2(df['TT4'])


# In[96]:


outliers3 =[] 
def detect_outliers3(column4):
    column5=sorted(column4)  #sorting the data
    Q1,Q3 = np.percentile(column5 , [25,75]) #finding 1st and 3rd quartile 
    IQR = Q3-Q1  
    lower_range = Q1-(1.5 * IQR)
    upper_range = Q3+(1.5 * IQR)
    for x in column5: #for each record in specified column
        if ((x> upper_range) or (x<lower_range)):
             outliers3.append(x)   #appending to list
    print(outliers3)   
outlier3=detect_outliers3(df['T4U'])


# In[97]:


outliers4=[] 
def detect_outliers4(column6):
    column7=sorted(column6)  #sorting the data
    Q1,Q3 = np.percentile(column7 , [25,75]) #finding 1st and 3rd quartile 
    IQR = Q3-Q1  
    lower_range = Q1-(1.5 * IQR)
    upper_range = Q3+(1.5 * IQR)
    for x in column7: #for each record in specified column
        if ((x> upper_range) or (x<lower_range)):
             outliers4.append(x)   #appending to list
    print(outliers4)   
outlier4=detect_outliers4(df['FTI'])


# In[98]:


outliers5 =[] 
def detect_outliers5(column8):
    column9=sorted(column8)  #sorting the data
    Q1,Q3 = np.percentile(column9 , [25,75]) #finding 1st and 3rd quartile 
    IQR = Q3-Q1  
    lower_range = Q1-(1.5 * IQR)
    upper_range = Q3+(1.5 * IQR)
    for x in column9: #for each record in specified column
        if ((x> upper_range) or (x<lower_range)):
             outliers5.append(x)   #appending to list
    print(outliers5)   
outlier5=detect_outliers5(df['age'])


# I decided to keep the outliers since they show important variability in the data. It is possible that patients had extreme values for these hormone levels, they do not seem like data entry errors. Getting rid of the outliers may increase our model's accuracy but they are essential to show the natural variation in medical data.

# In[165]:


#Countplot for patients without vs with hypothyroid
count,ax = plt.subplots()
ax = sns.countplot("Outcome", data = df)
ax.set_title (" Count of patients without vs with hypothyroid")
ax.set_xlabel ("Outcome")
ax.set_ylabel ("Frequency")
plt.show ()
df['Outcome'].value_counts()
#More number of patients were detected with hypothyroid.


# In[44]:


#Analyzing hormone levels for patients with and without hyperthyroid based on if they are sick or not.


# In[45]:


#fig, ax =plt.subplots(2,2)
sns.catplot(x="sick",y="T3", hue="Outcome", kind="bar",data=df)
#T3 levels were higher in those who were not sick. Hypothyroid patients had comparatively higher T3 levels.


# In[102]:


sns.catplot(x="sick",y="TT4", hue="Outcome", kind="bar", data=df)
#Hypothyroid patients had higher TT4 levels. 


# In[103]:


sns.catplot(x="sick",y="T4U", hue="Outcome", kind="bar", data=df)
#T4U levels were higher for patients who were not sick. For those who were not sick, patients without hypothyroid had higher T4U levels. For those who were sick, patients with hypothyroid had higher T4U levels.


# In[48]:


sns.catplot(x="sick",y="FTI", hue="Outcome", kind="bar", data=df)
#HYpothyroid patients had higher FTI levels.


# In[49]:


sns.catplot(x="sick",y="TSH", hue="Outcome", kind="bar", data=df)
#Healthy patients had higher TSH levels. These wre highest for those who were not sick.


# In[50]:


sns.catplot(x="tumor",y="T3", hue="Outcome", kind="box", data=df)
#Patients with tumor had higher T3 levels. Patients with hypothyroid had slightly higher T3 levels.


# In[51]:


sns.catplot(x="tumor",y="TT4", hue="Outcome", kind="box", data=df)
#Patients with hypothyroid had higher TT4 levels as comapred to those who did not have hypothyroid.


# In[52]:


sns.catplot(x="tumor",y="T4U", hue="Outcome", kind="box", data=df)
#Patients with a tumor had slightly higher T4U level.


# In[105]:


sns.catplot(x="tumor",y="FTI", hue="Outcome", kind="violin", data=df)
#Non-hypothyroid patients with a tumor had larger variability in FTI levels. In general, hypothyroid patients had higher FTI levels.


# In[106]:


sns.catplot(x="tumor",y="age", hue="Outcome", kind="box", data=df)
#For patients without a tumor, median age was approximate the same, irrespective of if they had hypothyroid or not. For patients with a tumor, age was slightly higher for patients with hypothyroid.


# In[107]:


num_bins=100
plt.hist(df1['age'], num_bins, density = 1, color ='gray')
plt.xlim([0, 100])
plt.title("Histogram for ages of patients")
plt.xlabel("Ages")
plt.ylabel("Density")
plt.show()  
#Highest number of patients were between 50-60 years old.


# In[108]:


#Correlation matrix for numeric attributes 
df_numeric=df[['age','T3','TT4','T4U',"TSH",'FTI']] #Pearson correlation between each pair of features
print(df_numeric.corr())


# In[109]:


#Plotting the correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(data=df_numeric.corr(),yticklabels=True,annot=True)


# In[110]:


#Strongest correlation-TT4, FTI
sns.set_style('whitegrid')
figure=plt.figure()
plt.xlabel("TT4")
plt.ylabel("FTI")
plt.title("Relationship between TT4 and FTI")
sns.scatterplot(x='TT4',y='FTI',data=df,color ='black',hue='Outcome',style="psych",alpha=0.4,palette="bright")
#Positive slope, as TT4 levels increase, so do FTI levels.


# In[111]:


sns.set_style('whitegrid')
figure=plt.figure()
plt.xlabel("TT4")
plt.ylabel("T3")
plt.title("Relationship between TT4 and T3")
sns.scatterplot(x='TT4',y='T3',data=df,color ='black',hue='Outcome',style="psych",alpha=0.4,palette="bright")
#Positive slope, as TT4 levels increase, so do T3 levels.


# In[141]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X = df[['age', 'sex', 'on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values
scaler = StandardScaler()
X=scaler.fit_transform(X) 
y = df['Outcome'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state = 3, stratify= y)
df.head()


# In[113]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
lor = LogisticRegression()
lor.fit(X_train,y_train)
y_pred = lor.predict(X_test)
lor_accuracy=round(accuracy_score(y_test,y_pred),4)*100
print("Accuracy for LOR: {}%".format(lor_accuracy))


# In[114]:


cm=confusion_matrix(y_test,y_pred)
print(cm)
TP=cm[0,0]
print("True positives are", TP)
FN=cm[0,1]
print("False negatives are", FN)
FP=cm[1,0]
print("False positives are", FP)
TN=cm[1,1]
print("True negatives are", TN)
TPR=TP/(TP + FN)
print("True positive rate is", np.round(TPR*100,2))
TNR=TN/(TN + FP)
print("True negative rate is", np.round(TNR*100,2))


# In[115]:


#Plotting the confusion matrix
plt.figure(figsize=(10,7))
p = sns.heatmap(cm, annot=True, cmap="Reds", fmt='g')
plt.title('Confusion matrix for LoR')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[116]:


X1 = df[['sex', 'on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X1=scaler.fit_transform(X1) 
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.50, random_state = 3, stratify= y) #50/50 split

lor.fit(X1_train,y_train)
lor_pred_without_age=lor.predict(X1_test)

lor_score_without_age=round(accuracy_score(y_test,lor_pred_without_age),4)*100
print("LoR Accuracy without age: {}%".format(lor_score_without_age))

X2 = df[['age', 'on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X2=scaler.fit_transform(X2) 
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.50, random_state = 3, stratify= y) #50/50 split

lor.fit(X2_train,y_train)
lor_pred_without_sex=lor.predict(X2_test)

lor_score_without_sex=round(accuracy_score(y_test,lor_pred_without_sex),4)*100
print("LoR Accuracy without sex: {}%".format(lor_score_without_sex))

X3 = df[['age', 'sex', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X3=scaler.fit_transform(X3) 
X3_train, X3_test, y_train, y_test = train_test_split(X3, y, test_size=0.50, random_state = 3, stratify= y) #50/50 split

lor.fit(X3_train,y_train)
lor_pred_without_onthyroxine=lor.predict(X3_test)

lor_score_without_onthyroxine=round(accuracy_score(y_test,lor_pred_without_onthyroxine),4)*100
print("LoR Accuracy without onthyroxine: {}%".format(lor_score_without_onthyroxine))


X4 = df[['age', 'sex', 'on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X4=scaler.fit_transform(X4) 
X4_train, X4_test, y_train, y_test = train_test_split(X4, y, test_size=0.50, random_state = 3, stratify= y) #50/50 split

lor.fit(X4_train,y_train)
lor_pred_without_queryonthyroxine=lor.predict(X4_test)

lor_score_without_queryonthyroxine=round(accuracy_score(y_test,lor_pred_without_queryonthyroxine),4)*100
print("LoR Accuracy without query on thyroxine: {}%".format(lor_score_without_queryonthyroxine))

X5 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X5=scaler.fit_transform(X5) 
X5_train, X5_test, y_train, y_test = train_test_split(X5, y, test_size=0.50, random_state = 3, stratify= y) #50/50 split

lor.fit(X5_train,y_train)
lor_pred_without_medication=lor.predict(X5_test)

lor_score_without_medication=round(accuracy_score(y_test,lor_pred_without_medication),4)*100
print("LoR Accuracy without medication: {}%".format(lor_score_without_medication))

X6 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'on antithyroid medication', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X6=scaler.fit_transform(X6) 
X6_train, X6_test, y_train, y_test = train_test_split(X6, y, test_size=0.50, random_state = 3, stratify= y) #50/50 split

lor.fit(X6_train,y_train)
lor_pred_without_sick=lor.predict(X6_test)

lor_score_without_sick=round(accuracy_score(y_test,lor_pred_without_sick),4)*100
print("LoR Accuracy without sick: {}%".format(lor_score_without_sick))

X7 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X7=scaler.fit_transform(X7) 
X7_train, X7_test, y_train, y_test = train_test_split(X7, y, test_size=0.50, random_state = 3, stratify= y) 

lor.fit(X7_train,y_train)
lor_pred_without_pregnant=lor.predict(X7_test)

lor_score_without_pregnant=round(accuracy_score(y_test,lor_pred_without_pregnant),4)*100
print("LoR Accuracy without pregnant: {}%".format(lor_score_without_pregnant))

X8 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X8=scaler.fit_transform(X8) 
X8_train, X8_test, y_train, y_test = train_test_split(X8, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X8_train,y_train)
lor_pred_without_surgery=lor.predict(X8_test)

lor_score_without_surgery=round(accuracy_score(y_test,lor_pred_without_surgery),4)*100
print("LoR Accuracy without surgery: {}%".format(lor_score_without_surgery))

X9 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X9=scaler.fit_transform(X9) 
X9_train, X9_test, y_train, y_test = train_test_split(X9, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X9_train,y_train)
lor_pred_without_I131treatment=lor.predict(X9_test)

lor_score_without_I131treatment=round(accuracy_score(y_test,lor_pred_without_I131treatment),4)*100
print("LoR Accuracy without I131 treatment: {}%".format(lor_score_without_I131treatment))

X10 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X10=scaler.fit_transform(X10) 
X10_train, X10_test, y_train, y_test = train_test_split(X10, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X10_train,y_train)
lor_pred_without_queryhypothyroid=lor.predict(X10_test)

lor_score_without_queryhypothyroid=round(accuracy_score(y_test,lor_pred_without_queryhypothyroid),4)*100
print("LoR Accuracy without query hypothyroid: {}%".format(lor_score_without_queryhypothyroid))

X11 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X11=scaler.fit_transform(X11) 
X11_train, X11_test, y_train, y_test = train_test_split(X11, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X11_train,y_train)
lor_pred_without_queryhyperthyroid=lor.predict(X11_test)

lor_score_without_queryhyperthyroid=round(accuracy_score(y_test,lor_pred_without_queryhyperthyroid),4)*100
print("LoR Accuracy without query hyperthyroid: {}%".format(lor_score_without_queryhyperthyroid))

X12 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X12=scaler.fit_transform(X12) 
X12_train, X12_test, y_train, y_test = train_test_split(X12, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X12_train,y_train)
lor_pred_without_lithium=lor.predict(X12_test)

lor_score_without_lithium=round(accuracy_score(y_test,lor_pred_without_lithium),4)*100
print("LoR Accuracy without lithium: {}%".format(lor_score_without_lithium))

X13 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X13=scaler.fit_transform(X13) 
X13_train, X13_test, y_train, y_test = train_test_split(X13, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X13_train,y_train)
lor_pred_without_goitre=lor.predict(X13_test)

lor_score_without_goitre=round(accuracy_score(y_test,lor_pred_without_goitre),4)*100
print("LoR Accuracy without goitre: {}%".format(lor_score_without_goitre))

X14 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'hypopituitary', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X14=scaler.fit_transform(X14) 
X14_train, X14_test, y_train, y_test = train_test_split(X14, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X14_train,y_train)
lor_pred_without_tumor=lor.predict(X14_test)

lor_score_without_tumor=round(accuracy_score(y_test,lor_pred_without_tumor),4)*100
print("LoR Accuracy without tumor: {}%".format(lor_score_without_tumor))

X15 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'psych', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X15=scaler.fit_transform(X15) 
X15_train, X15_test, y_train, y_test = train_test_split(X15, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X15_train,y_train)
lor_pred_without_hypopituitary=lor.predict(X15_test)

lor_score_without_hypopituitary=round(accuracy_score(y_test,lor_pred_without_hypopituitary),4)*100
print("LoR Accuracy without hypopituitary: {}%".format(lor_score_without_hypopituitary))

X16 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'TSH measured', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X16=scaler.fit_transform(X16) 
X16_train, X16_test, y_train, y_test = train_test_split(X16, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X16_train,y_train)
lor_pred_without_psych=lor.predict(X16_test)

lor_score_without_psych=round(accuracy_score(y_test,lor_pred_without_psych),4)*100
print("LoR Accuracy without psych: {}%".format(lor_score_without_psych))

X17 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X17=scaler.fit_transform(X17) 
X17_train, X17_test, y_train, y_test = train_test_split(X17, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X17_train,y_train)
lor_pred_without_TSHmeasured=lor.predict(X17_test)

lor_score_without_TSHmeasured=round(accuracy_score(y_test,lor_pred_without_TSHmeasured),4)*100
print("LoR Accuracy without TSH measured: {}%".format(lor_score_without_TSHmeasured))

X18 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X18=scaler.fit_transform(X18) 
X18_train, X18_test, y_train, y_test = train_test_split(X18, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X18_train,y_train)
lor_pred_without_TSH=lor.predict(X18_test)

lor_score_without_TSH=round(accuracy_score(y_test,lor_pred_without_TSH),4)*100
print("LoR Accuracy without TSH: {}%".format(lor_score_without_TSH))

X19 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X19=scaler.fit_transform(X19) 
X19_train, X19_test, y_train, y_test = train_test_split(X19, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X19_train,y_train)
lor_pred_without_T3measured=lor.predict(X19_test)

lor_score_without_T3measured=round(accuracy_score(y_test,lor_pred_without_T3measured),4)*100
print("LoR Accuracy without T3 measured: {}%".format(lor_score_without_T3measured))

X20 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3 measured', 'TT4 measured', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X20=scaler.fit_transform(X20) 
X20_train, X20_test, y_train, y_test = train_test_split(X20, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X20_train,y_train)
lor_pred_without_T3=lor.predict(X20_test)

lor_score_without_T3=round(accuracy_score(y_test,lor_pred_without_T3),4)*100
print("LoR Accuracy without T3: {}%".format(lor_score_without_T3))

X21 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3 measured', 'T3', 'TT4', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X21=scaler.fit_transform(X21) 
X21_train, X21_test, y_train, y_test = train_test_split(X21, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X21_train,y_train)
lor_pred_without_TT4measured=lor.predict(X21_test)

lor_score_without_TT4measured=round(accuracy_score(y_test,lor_pred_without_TT4measured),4)*100
print("LoR Accuracy without TT4 measured: {}%".format(lor_score_without_TT4measured))

X22 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3 measured', 'T3', 'TT4 measured', 'T4U measured', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X22=scaler.fit_transform(X22) 
X22_train, X22_test, y_train, y_test = train_test_split(X22, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X22_train,y_train)
lor_pred_without_TT4=lor.predict(X22_test)

lor_score_without_TT4=round(accuracy_score(y_test,lor_pred_without_TT4),4)*100
print("LoR Accuracy without TT4: {}%".format(lor_score_without_TT4))

X23 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U',
       'FTI measured', 'FTI', 'TBG measured']].values 

X23=scaler.fit_transform(X23) 
X23_train, X23_test, y_train, y_test = train_test_split(X23, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X23_train,y_train)
lor_pred_without_T4Umeasured=lor.predict(X23_test)

lor_score_without_T4Umeasured=round(accuracy_score(y_test,lor_pred_without_T4Umeasured),4)*100
print("LoR Accuracy without T4U measured: {}%".format(lor_score_without_T4Umeasured))

X24 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured',
       'FTI measured', 'FTI', 'TBG measured']].values 

X24=scaler.fit_transform(X24) 
X24_train, X24_test, y_train, y_test = train_test_split(X24, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X24_train,y_train)
lor_pred_without_T4U=lor.predict(X24_test)

lor_score_without_T4U=round(accuracy_score(y_test,lor_pred_without_T4U),4)*100
print("LoR Accuracy without T4U: {}%".format(lor_score_without_T4U))

X25 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured',
       'T4U', 'FTI', 'TBG measured']].values 

X25=scaler.fit_transform(X25) 
X25_train, X25_test, y_train, y_test = train_test_split(X25, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X25_train,y_train)
lor_pred_without_FTImeasured=lor.predict(X25_test)

lor_score_without_FTImeasured=round(accuracy_score(y_test,lor_pred_without_FTImeasured),4)*100
print("LoR Accuracy without FTI measured: {}%".format(lor_score_without_FTImeasured))

X26 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured',
       'T4U', 'FTI measured', 'TBG measured']].values 

X26=scaler.fit_transform(X26) 
X26_train, X26_test, y_train, y_test = train_test_split(X26, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X26_train,y_train)
lor_pred_without_FTI=lor.predict(X26_test)

lor_score_without_FTI=round(accuracy_score(y_test,lor_pred_without_FTI),4)*100
print("LoR Accuracy without FTI: {}%".format(lor_score_without_FTI))

X27 = df[['age', 'sex', 'query on thyroxine',
       'on thyroxine', 'sick', 'on antithyroid medication', 'pregnant',
       'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hypothyroid',
       'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'TSH', 'T3 measured', 'T3', 'TT4 measured', 'TT4', 'T4U measured',
       'T4U', 'FTI measured', 'FTI']].values 

X27=scaler.fit_transform(X27) 
X27_train, X27_test, y_train, y_test = train_test_split(X27, y, test_size=0.50, random_state = 3, stratify= y) 
lor.fit(X27_train,y_train)
lor_pred_without_TBGmeasured=lor.predict(X27_test)

lor_score_without_TBGmeasured=round(accuracy_score(y_test,lor_pred_without_TBGmeasured),4)*100
print("LoR Accuracy without TBG measured: {}%".format(lor_score_without_TBGmeasured))


# In[117]:


from sklearn.neighbors import KNeighborsClassifier
model_data=[]
for k in range(1,13,2): #for each value of k
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred2 = knn.predict(X_train)  
    model_data.append([k, accuracy_score(y_pred2, y_train)]) 
    
model_data_frame = pd.DataFrame(columns=["K", "Accuracy_Score"], data=model_data)
print(model_data_frame.sort_values("Accuracy_Score")) 


# In[118]:


#Manhattan distance or L1 norm
model_data2=[]
for k2 in range(1,13,2): 
    knn2=KNeighborsClassifier(n_neighbors=k2,p=1)
    knn2.fit(X_train, y_train)
    y_pred2b = knn2.predict(X_train)  
    model_data2.append([k2, accuracy_score(y_pred2b, y_train)]) 
    
model_data_frame2 = pd.DataFrame(columns=["K", "Accuracy_Score"], data=model_data2)
print(model_data_frame2.sort_values("Accuracy_Score")) 


# In[119]:


def minkowski_p(a,b,p): 
    return np.linalg.norm(a-b, ord=p)


# In[120]:


p=1.5
model_data3=[]
for k3 in range(1,13,2): 
    knn3=KNeighborsClassifier(n_neighbors=k3,metric = lambda a,b: minkowski_p(a,b,p)) 
    knn3.fit(X_train, y_train) 
    pred2c = knn3.predict(X_train) 
    model_data3.append([k3, accuracy_score(pred2c,y_train)]) 
    
model_data_frame3 = pd.DataFrame(columns=["K", "Accuracy_Score"], data=model_data3) 
print(model_data_frame3.sort_values("Accuracy_Score")) 


# In[ ]:


model_data4=[]
for k4 in range(1,13,2): 
    knn4=KNeighborsClassifier(n_neighbors=k4,weights="distance") 
    knn4.fit(X_train, np.ravel(y_train)) 
    pred2d = knn4.predict(X_train) 
    model_data4.append([k4, accuracy_score(pred2d,y_train)]) 
    
model_data_frame4 = pd.DataFrame(columns=["K", "Accuracy_Score"], data=model_data4) 
print(model_data_frame4.sort_values("Accuracy_Score")) 


# In[ ]:


#Finding centroid of each class and assigning label based on nearest centroid
from sklearn.neighbors import NearestCentroid
knn5=NearestCentroid() 
knn5.fit(X_train, np.ravel(y_train)) 
pred2e = knn5.predict(X_train) 
score_2e=round(accuracy_score(y_test,pred2e),4)*100
print("KNN Accuracy with nearest centroid method: {}%".format(score_2e))


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
sns.set_style('whitegrid')
figure=plt.figure()
figure.figsize =(10 ,4)
ax = plt.gca ()
ax.xaxis.set_major_locator(MaxNLocator(integer = True))
plt.plot(range (1 ,13 ,2),[1,0.96,0.95,0.944,0.942,0.939], color ='black', linestyle ='dashed',marker ='o', markerfacecolor ='black', markersize =10,label="Regular KNN (L2 norm)")
plt.plot(range (1 ,13 ,2),[1.0,0.96,0.95,0.949,0.944,0.940], color ='blue', linestyle ='dashed',marker ='o', markerfacecolor ='blue', markersize =10,label="L1 norm")
plt.plot(range (1 ,13 ,2),[1.0,0.96,0.952,0.945,0.942,0.940], color ='red', linestyle ='dashed',marker ='o', markerfacecolor ='red', markersize =10,label="Minkowski distance")
plt.plot(range (1 ,13 ,2),[1.0,1.0,1.0,1.0,1.0,1.0], color ='gray', linestyle ='dashed',marker ='o', markerfacecolor ='gray', markersize =10,label="Weighted KNN")
plt.title('Best k value for different KNN variations')
plt.xlabel('k neighbors')
plt.ylabel('Accuracy')
plt.legend()


# In[ ]:


#k=3, weighted knn perform the best.


# In[140]:


knn2=KNeighborsClassifier(n_neighbors=3,weights="distance")
knn2.fit(X_train, np.ravel(y_train))
y_pred3 = knn2.predict(X_test)
score2 = round(accuracy_score(y_test, y_pred3),4)*100
print("Accuracy for KNN (k=3, weighted knn): {}%".format(score2))


# In[ ]:


cm2=confusion_matrix(y_test,y_pred3)
print(cm2)
TP2=cm2[0,0]
print("True positives are", TP2)
FN2=cm2[0,1]
print("False negatives are", FN2)
FP2=cm2[1,0]
print("False positives are", FP2)
TN2=cm2[1,1]
print("True negatives are", TN2)
TPR2=TP2/(TP2 + FN2)
print("True positive rate is", np.round(TPR2*100,2))
TNR2=TN2/(TN2 + FP2)
print("True negative rate is", np.round(TNR2*100,2))


# In[ ]:


#Plotting the confusion matrix
plt.figure(figsize=(10,7))
p = sns.heatmap(cm2, annot=True, cmap="Reds", fmt='g')
plt.title('Confusion matrix for KNN')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[ ]:


knn2.fit(X1_train,y_train)
knn2_pred_without_age=knn2.predict(X1_test)

knn2_score_without_age=round(accuracy_score(y_test,knn2_pred_without_age),4)*100
print("knn Accuracy without age: {}%".format(knn2_score_without_age))

knn2.fit(X2_train,y_train)
knn2_pred_without_sex=knn2.predict(X2_test)

knn2_score_without_sex=round(accuracy_score(y_test,knn2_pred_without_sex),4)*100
print("knn Accuracy without sex: {}%".format(knn2_score_without_sex))

knn2.fit(X3_train,y_train)
knn2_pred_without_onthyroxine=knn2.predict(X3_test)

knn2_score_without_onthyroxine=round(accuracy_score(y_test,knn2_pred_without_onthyroxine),4)*100
print("knn Accuracy without onthyroxine: {}%".format(knn2_score_without_onthyroxine))


knn2.fit(X4_train,y_train)
knn2_pred_without_queryonthyroxine=knn2.predict(X4_test)

knn2_score_without_queryonthyroxine=round(accuracy_score(y_test,knn2_pred_without_queryonthyroxine),4)*100
print("knn Accuracy without query on thyroxine: {}%".format(knn2_score_without_queryonthyroxine))

knn2.fit(X5_train,y_train)
knn2_pred_without_medication=knn2.predict(X5_test)

knn2_score_without_medication=round(accuracy_score(y_test,knn2_pred_without_medication),4)*100
print("knn Accuracy without medication: {}%".format(knn2_score_without_medication))


knn2.fit(X6_train,y_train)
knn2_pred_without_sick=knn2.predict(X6_test)

knn2_score_without_sick=round(accuracy_score(y_test,knn2_pred_without_sick),4)*100
print("knn Accuracy without sick: {}%".format(knn2_score_without_sick))

knn2.fit(X7_train,y_train)
knn2_pred_without_pregnant=knn2.predict(X7_test)

knn2_score_without_pregnant=round(accuracy_score(y_test,knn2_pred_without_pregnant),4)*100
print("knn2 Accuracy without pregnant: {}%".format(knn2_score_without_pregnant))

 
knn2.fit(X8_train,y_train)
knn2_pred_without_surgery=knn2.predict(X8_test)

knn2_score_without_surgery=round(accuracy_score(y_test,knn2_pred_without_surgery),4)*100
print("knn2 Accuracy without surgery: {}%".format(knn2_score_without_surgery))

knn2.fit(X9_train,y_train)
knn2_pred_without_I131treatment=knn2.predict(X9_test)

knn2_score_without_I131treatment=round(accuracy_score(y_test,knn2_pred_without_I131treatment),4)*100
print("knn2 Accuracy without I131 treatment: {}%".format(knn2_score_without_I131treatment))

knn2.fit(X10_train,y_train)
knn2_pred_without_queryhypothyroid=knn2.predict(X10_test)

knn2_score_without_queryhypothyroid=round(accuracy_score(y_test,knn2_pred_without_queryhypothyroid),4)*100
print("knn2 Accuracy without query hypothyroid: {}%".format(knn2_score_without_queryhypothyroid))


knn2.fit(X11_train,y_train)
knn2_pred_without_queryhyperthyroid=knn2.predict(X11_test)

knn2_score_without_queryhyperthyroid=round(accuracy_score(y_test,knn2_pred_without_queryhyperthyroid),4)*100
print("knn2 Accuracy without query hyperthyroid: {}%".format(knn2_score_without_queryhyperthyroid))

knn2.fit(X12_train,y_train)
knn2_pred_without_lithium=knn2.predict(X12_test)

knn2_score_without_lithium=round(accuracy_score(y_test,knn2_pred_without_lithium),4)*100
print("knn2 Accuracy without lithium: {}%".format(knn2_score_without_lithium))

knn2.fit(X13_train,y_train)
knn2_pred_without_goitre=knn2.predict(X13_test)

knn2_score_without_goitre=round(accuracy_score(y_test,knn2_pred_without_goitre),4)*100
print("knn2 Accuracy without goitre: {}%".format(knn2_score_without_goitre))

knn2.fit(X14_train,y_train)
knn2_pred_without_tumor=knn2.predict(X14_test)

knn2_score_without_tumor=round(accuracy_score(y_test,knn2_pred_without_tumor),4)*100
print("knn2 Accuracy without tumor: {}%".format(knn2_score_without_tumor))

knn2.fit(X15_train,y_train)
knn2_pred_without_hypopituitary=knn2.predict(X15_test)

knn2_score_without_hypopituitary=round(accuracy_score(y_test,knn2_pred_without_hypopituitary),4)*100
print("knn2 Accuracy without hypopituitary: {}%".format(knn2_score_without_hypopituitary))

knn2.fit(X16_train,y_train)
knn2_pred_without_psych=knn2.predict(X16_test)

knn2_score_without_psych=round(accuracy_score(y_test,knn2_pred_without_psych),4)*100
print("knn2 Accuracy without psych: {}%".format(knn2_score_without_psych))

knn2.fit(X17_train,y_train)
knn2_pred_without_TSHmeasured=knn2.predict(X17_test)

knn2_score_without_TSHmeasured=round(accuracy_score(y_test,knn2_pred_without_TSHmeasured),4)*100
print("knn2 Accuracy without TSH measured: {}%".format(knn2_score_without_TSHmeasured))

knn2.fit(X18_train,y_train)
knn2_pred_without_TSH=knn2.predict(X18_test)

knn2_score_without_TSH=round(accuracy_score(y_test,knn2_pred_without_TSH),4)*100
print("knn2 Accuracy without TSH: {}%".format(knn2_score_without_TSH))

knn2.fit(X19_train,y_train)
knn2_pred_without_T3measured=knn2.predict(X19_test)

knn2_score_without_T3measured=round(accuracy_score(y_test,knn2_pred_without_T3measured),4)*100
print("knn2 Accuracy without T3 measured: {}%".format(knn2_score_without_T3measured))

knn2.fit(X20_train,y_train)
knn2_pred_without_T3=knn2.predict(X20_test)

knn2_score_without_T3=round(accuracy_score(y_test,knn2_pred_without_T3),4)*100
print("knn2 Accuracy without T3: {}%".format(knn2_score_without_T3))

knn2.fit(X21_train,y_train)
knn2_pred_without_TT4measured=knn2.predict(X21_test)

knn2_score_without_TT4measured=round(accuracy_score(y_test,knn2_pred_without_TT4measured),4)*100
print("knn2 Accuracy without TT4 measured: {}%".format(knn2_score_without_TT4measured))

knn2.fit(X22_train,y_train)
knn2_pred_without_TT4=knn2.predict(X22_test)

knn2_score_without_TT4=round(accuracy_score(y_test,knn2_pred_without_TT4),4)*100
print("knn2 Accuracy without TT4: {}%".format(knn2_score_without_TT4))

knn2.fit(X23_train,y_train)
knn2_pred_without_T4Umeasured=knn2.predict(X23_test)

knn2_score_without_T4Umeasured=round(accuracy_score(y_test,knn2_pred_without_T4Umeasured),4)*100
print("knn2 Accuracy without T4U measured: {}%".format(knn2_score_without_T4Umeasured))

knn2.fit(X24_train,y_train)
knn2_pred_without_T4U=knn2.predict(X24_test)

knn2_score_without_T4U=round(accuracy_score(y_test,knn2_pred_without_T4U),4)*100
print("knn2 Accuracy without T4U: {}%".format(knn2_score_without_T4U))

knn2.fit(X25_train,y_train)
knn2_pred_without_FTImeasured=knn2.predict(X25_test)

knn2_score_without_FTImeasured=round(accuracy_score(y_test,knn2_pred_without_FTImeasured),4)*100
print("knn2 Accuracy without FTI measured: {}%".format(knn2_score_without_FTImeasured))

knn2.fit(X26_train,y_train)
knn2_pred_without_FTI=knn2.predict(X26_test)

knn2_score_without_FTI=round(accuracy_score(y_test,knn2_pred_without_FTI),4)*100
print("knn2 Accuracy without FTI: {}%".format(knn2_score_without_FTI))

knn2.fit(X27_train,y_train)
knn2_pred_without_TBGmeasured=knn2.predict(X27_test)

knn2_score_without_TBGmeasured=round(accuracy_score(y_test,knn2_pred_without_TBGmeasured),4)*100
print("knn2 Accuracy without TBG measured: {}%".format(knn2_score_without_TBGmeasured))


# In[ ]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB
X_cont=df[['age','T3','TT4','T4U','FTI']].values
X_cat=df[['sex','on thyroxine', 'query on thyroxine',
       'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
       'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured',
       'T3 measured', 'TT4 measured', 'T4U measured', 
       'FTI measured', 'TBG measured']].values
#X_cont_train, X_cont_test, y_train, y_test = train_test_split(X_cont, y, test_size=0.50, random_state = 3, stratify= y)
NB_classifier1 = GaussianNB().fit(X_cont,np.ravel(y))  
#X_cat_train, X_cat_test, y_train, y_test = train_test_split(X_cat, y, test_size=0.50, random_state = 3, stratify= y)
NB_classifier2 = MultinomialNB().fit(X_cat,np.ravel(y)) 
y_pred_cont=NB_classifier1.predict_proba(X_cont)
NB_classifier2.predict_proba(X_cat)
y_pred_cat=NB_classifier2.predict_proba(X_cat)
mixed_X=np.hstack((y_pred_cont,y_pred_cat)) 
X_train1, X_test1, y_train1, y_test1 = train_test_split(mixed_X, y, test_size=0.50, random_state = 3, stratify= y)
NB_classifier3=GaussianNB().fit(X_train1,np.ravel(y_train1)) 
y_pred_mixed = NB_classifier3.predict(X_test1)
score4 = round(accuracy_score(y_test1, y_pred_mixed),4)*100
print("Accuracy for Naive Bayes: {}%".format(score4))
cm3=confusion_matrix(y_test1,y_pred_mixed)
print(cm3)
TP3=cm3[0,0]
print("True positives are", TP3)
FN3=cm3[0,1]
print("False negatives are", FN3)
FP3=cm3[1,0]
print("False positives are", FP3)
TN3=cm3[1,1]
print("True negatives are", TN3)
TPR3=TP3/(TP3 + FN3)
print("True positive rate is", np.round(TPR3*100,2))
TNR3=TN3/(TN3 + FP3)
print("True negative rate is", np.round(TNR3*100,2))
plt.figure(figsize=(10,7))
p = sns.heatmap(cm3, annot=True, cmap="Reds", fmt='g')
plt.title('Confusion matrix for Naive Bayes')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[131]:


from sklearn.ensemble import RandomForestClassifier
model_data4=[]
for N in range(1,11,2): 
    rfa=RandomForestClassifier(criterion='entropy',max_depth=1,n_estimators=N) 
    rfa.fit(X_train, y_train)
    y_pred4a = rfa.predict(X_test) 
    model_data4.append([N, 1-accuracy_score(y_test,y_pred4a)]) 
    
model_data_frame4 = pd.DataFrame(columns=["N", "Error rate"], data=model_data4) 
print(model_data_frame4.sort_values("Error rate")) 


# In[132]:


model_data4b=[]
for Nb in range(1,11,2): 
    rfb=RandomForestClassifier(criterion='entropy',max_depth=2,n_estimators=Nb) 
    rfb.fit(X_train, y_train)
    y_pred4b = rfb.predict(X_test) 
    model_data4b.append([Nb, 1-accuracy_score(y_test,y_pred4b)]) 
    
model_data_frame4b = pd.DataFrame(columns=["N", "Error rate"], data=model_data4b) 
print(model_data_frame4b.sort_values("Error rate")) 


# In[133]:


model_data4c=[]
for Nc in range(1,11,2): 
    rfc=RandomForestClassifier(criterion='entropy',max_depth=3,n_estimators=Nc) 
    rfc.fit(X_train, y_train)
    y_pred4c = rfc.predict(X_test) 
    model_data4c.append([Nc, 1-accuracy_score(y_test,y_pred4c)]) 
    
model_data_frame4c = pd.DataFrame(columns=["N", "Error rate"], data=model_data4c) 
print(model_data_frame4c.sort_values("Error rate")) 


# In[134]:


model_data4d=[]
for Nd in range(1,11,2): 
    rfd=RandomForestClassifier(criterion='entropy',max_depth=4,n_estimators=Nd) 
    rfd.fit(X_train, y_train)
    y_pred4d = rfd.predict(X_test) 
    model_data4d.append([Nd, 1-accuracy_score(y_test,y_pred4d)]) 
    
model_data_frame4d = pd.DataFrame(columns=["N", "Error rate"], data=model_data4d) 
print(model_data_frame4d.sort_values("Error rate")) 


# In[135]:


model_data4e=[]
for Ne in range(1,11,2): 
    rfe=RandomForestClassifier(criterion='entropy',max_depth=5,n_estimators=Ne) 
    rfe.fit(X_train, y_train)
    y_pred4e = rfe.predict(X_test) 
    model_data4e.append([Ne, 1-accuracy_score(y_test,y_pred4e)]) 
    
model_data_frame4e = pd.DataFrame(columns=["N", "Error rate"], data=model_data4e) 
print(model_data_frame4e.sort_values("Error rate")) 


# In[136]:


sns.set_style('whitegrid')
figure=plt.figure()
figure.figsize =(10 ,4)

plt.plot(range (1 ,11 ,2),[0.019,0.030,0.076,0.077,0.077,], color ='gray', linestyle ='dashed',marker ='o', markerfacecolor ='gray', markersize =10,label="Depth=1")
plt.plot(range (1 ,11 ,2),[0.025,0.050,0.068,0.070,0.079], color ='blue', linestyle ='dashed',marker ='o', markerfacecolor ='blue', markersize =10,label="Depth=2")
plt.plot(range (1 ,11 ,2),[0.010,0.016,0.027,0.040,0.080], color ='red', linestyle ='dashed',marker ='o', markerfacecolor ='red', markersize =10,label="Depth=3")
plt.plot(range (1 ,11 ,2),[0.009,0.016,0.026,0.032,0.042], color ='green', linestyle ='dashed',marker ='o', markerfacecolor ='green', markersize =10,label="Depth=4")
plt.plot(range (1 ,11 ,2),[0.008,0.016,0.020,0.030,0.032], color ='black', linestyle ='dashed',marker ='o', markerfacecolor ='black', markersize =10,label="Depth=5")
plt.title('Best N and d value')
plt.xlabel('Number of decision trees/ weak classifiers/ base learners')
plt.ylabel('Error rate')
plt.legend()


# In[137]:


rf=RandomForestClassifier(criterion='entropy',max_depth=5,n_estimators=3) 
rf.fit(X_train, y_train)
y_pred4 = rf.predict(X_test) 
score4=round(accuracy_score(y_test,y_pred4),4)*100
print("Random Forest Accuracy(d=5,N=3): {}%".format(np.round(score4,2)))


# In[138]:


cm4=confusion_matrix(y_test,y_pred4)
print(cm4)
TP4=cm4[0,0]
print("True positives are", TP4)
FN4=cm4[0,1]
print("False negatives are", FN4)
FP4=cm4[1,0]
print("False positives are", FP4)
TN4=cm4[1,1]
print("True negatives are", TN4)
TPR4=TP4/(TP4 + FN4)
print("True positive rate is", np.round(TPR4*100,2))
TNR4=TN4/(TN4 + FP4)
print("True negative rate is", np.round(TNR4*100,2))
plt.figure(figsize=(10,7))
p = sns.heatmap(cm4, annot=True, cmap="Reds", fmt='g')
plt.title('Confusion matrix for Random Forest')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[139]:


rf.fit(X1_train,y_train)
rf_pred_without_age=rf.predict(X1_test)

rf_score_without_age=round(accuracy_score(y_test,rf_pred_without_age),4)*100
print("rf Accuracy without age: {}%".format(rf_score_without_age))

rf.fit(X2_train,y_train)
rf_pred_without_sex=rf.predict(X2_test)

rf_score_without_sex=round(accuracy_score(y_test,rf_pred_without_sex),4)*100
print("rf Accuracy without sex: {}%".format(rf_score_without_sex))

rf.fit(X3_train,y_train)
rf_pred_without_onthyroxine=rf.predict(X3_test)

rf_score_without_onthyroxine=round(accuracy_score(y_test,rf_pred_without_onthyroxine),4)*100
print("rf Accuracy without onthyroxine: {}%".format(rf_score_without_onthyroxine))


rf.fit(X4_train,y_train)
rf_pred_without_queryonthyroxine=rf.predict(X4_test)

rf_score_without_queryonthyroxine=round(accuracy_score(y_test,rf_pred_without_queryonthyroxine),4)*100
print("rf Accuracy without query on thyroxine: {}%".format(rf_score_without_queryonthyroxine))

rf.fit(X5_train,y_train)
rf_pred_without_medication=rf.predict(X5_test)

rf_score_without_medication=round(accuracy_score(y_test,rf_pred_without_medication),4)*100
print("rf Accuracy without medication: {}%".format(rf_score_without_medication))


rf.fit(X6_train,y_train)
rf_pred_without_sick=rf.predict(X6_test)

rf_score_without_sick=round(accuracy_score(y_test,rf_pred_without_sick),4)*100
print("rf Accuracy without sick: {}%".format(rf_score_without_sick))

rf.fit(X7_train,y_train)
rf_pred_without_pregnant=rf.predict(X7_test)

rf_score_without_pregnant=round(accuracy_score(y_test,rf_pred_without_pregnant),4)*100
print("rf Accuracy without pregnant: {}%".format(rf_score_without_pregnant))

 
rf.fit(X8_train,y_train)
rf_pred_without_surgery=rf.predict(X8_test)

rf_score_without_surgery=round(accuracy_score(y_test,rf_pred_without_surgery),4)*100
print("rf Accuracy without surgery: {}%".format(rf_score_without_surgery))

rf.fit(X9_train,y_train)
rf_pred_without_I131treatment=rf.predict(X9_test)

rf_score_without_I131treatment=round(accuracy_score(y_test,rf_pred_without_I131treatment),4)*100
print("rf Accuracy without I131 treatment: {}%".format(rf_score_without_I131treatment))

rf.fit(X10_train,y_train)
rf_pred_without_queryhypothyroid=rf.predict(X10_test)

rf_score_without_queryhypothyroid=round(accuracy_score(y_test,rf_pred_without_queryhypothyroid),4)*100
print("rf Accuracy without query hypothyroid: {}%".format(rf_score_without_queryhypothyroid))


rf.fit(X11_train,y_train)
rf_pred_without_queryhyperthyroid=rf.predict(X11_test)

rf_score_without_queryhyperthyroid=round(accuracy_score(y_test,rf_pred_without_queryhyperthyroid),4)*100
print("rf Accuracy without query hyperthyroid: {}%".format(rf_score_without_queryhyperthyroid))

rf.fit(X12_train,y_train)
rf_pred_without_lithium=rf.predict(X12_test)

rf_score_without_lithium=round(accuracy_score(y_test,rf_pred_without_lithium),4)*100
print("rf Accuracy without lithium: {}%".format(rf_score_without_lithium))

rf.fit(X13_train,y_train)
rf_pred_without_goitre=rf.predict(X13_test)

rf_score_without_goitre=round(accuracy_score(y_test,rf_pred_without_goitre),4)*100
print("rf Accuracy without goitre: {}%".format(rf_score_without_goitre))

rf.fit(X14_train,y_train)
rf_pred_without_tumor=rf.predict(X14_test)

rf_score_without_tumor=round(accuracy_score(y_test,rf_pred_without_tumor),4)*100
print("rf Accuracy without tumor: {}%".format(rf_score_without_tumor))

rf.fit(X15_train,y_train)
rf_pred_without_hypopituitary=rf.predict(X15_test)

rf_score_without_hypopituitary=round(accuracy_score(y_test,rf_pred_without_hypopituitary),4)*100
print("rf Accuracy without hypopituitary: {}%".format(rf_score_without_hypopituitary))

rf.fit(X16_train,y_train)
rf_pred_without_psych=rf.predict(X16_test)

rf_score_without_psych=round(accuracy_score(y_test,rf_pred_without_psych),4)*100
print("rf Accuracy without psych: {}%".format(rf_score_without_psych))

rf.fit(X17_train,y_train)
rf_pred_without_TSHmeasured=rf.predict(X17_test)

rf_score_without_TSHmeasured=round(accuracy_score(y_test,rf_pred_without_TSHmeasured),4)*100
print("rf Accuracy without TSH measured: {}%".format(rf_score_without_TSHmeasured))

rf.fit(X18_train,y_train)
rf_pred_without_TSH=rf.predict(X18_test)

rf_score_without_TSH=round(accuracy_score(y_test,rf_pred_without_TSH),4)*100
print("rf Accuracy without TSH: {}%".format(rf_score_without_TSH))

rf.fit(X19_train,y_train)
rf_pred_without_T3measured=rf.predict(X19_test)

rf_score_without_T3measured=round(accuracy_score(y_test,rf_pred_without_T3measured),4)*100
print("rf Accuracy without T3 measured: {}%".format(rf_score_without_T3measured))

rf.fit(X20_train,y_train)
rf_pred_without_T3=rf.predict(X20_test)

rf_score_without_T3=round(accuracy_score(y_test,rf_pred_without_T3),4)*100
print("rf Accuracy without T3: {}%".format(rf_score_without_T3))

rf.fit(X21_train,y_train)
rf_pred_without_TT4measured=rf.predict(X21_test)

rf_score_without_TT4measured=round(accuracy_score(y_test,rf_pred_without_TT4measured),4)*100
print("rf Accuracy without TT4 measured: {}%".format(rf_score_without_TT4measured))

rf.fit(X22_train,y_train)
rf_pred_without_TT4=rf.predict(X22_test)

rf_score_without_TT4=round(accuracy_score(y_test,rf_pred_without_TT4),4)*100
print("rf Accuracy without TT4: {}%".format(rf_score_without_TT4))

rf.fit(X23_train,y_train)
rf_pred_without_T4Umeasured=rf.predict(X23_test)

rf_score_without_T4Umeasured=round(accuracy_score(y_test,rf_pred_without_T4Umeasured),4)*100
print("rf Accuracy without T4U measured: {}%".format(rf_score_without_T4Umeasured))

rf.fit(X24_train,y_train)
rf_pred_without_T4U=rf.predict(X24_test)

rf_score_without_T4U=round(accuracy_score(y_test,rf_pred_without_T4U),4)*100
print("rf Accuracy without T4U: {}%".format(rf_score_without_T4U))

rf.fit(X25_train,y_train)
rf_pred_without_FTImeasured=rf.predict(X25_test)

rf_score_without_FTImeasured=round(accuracy_score(y_test,rf_pred_without_FTImeasured),4)*100
print("rf Accuracy without FTI measured: {}%".format(rf_score_without_FTImeasured))

rf.fit(X26_train,y_train)
rf_pred_without_FTI=rf.predict(X26_test)

rf_score_without_FTI=round(accuracy_score(y_test,rf_pred_without_FTI),4)*100
print("rf Accuracy without FTI: {}%".format(rf_score_without_FTI))

rf.fit(X27_train,y_train)
rf_pred_without_TBGmeasured=rf.predict(X27_test)

rf_score_without_TBGmeasured=round(accuracy_score(y_test,rf_pred_without_TBGmeasured),4)*100
print("rf Accuracy without TBG measured: {}%".format(rf_score_without_TBGmeasured))


# In[102]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion ='entropy')
dt.fit(X_train,np.ravel(y_train))
y_pred5=dt.predict(X_test)
score6 = round(accuracy_score(y_test, y_pred5),4)*100
print("Accuracy for Decision Tree: {}%".format(score6))

cm5=confusion_matrix(y_test,y_pred5)
print(cm5)
TP5=cm5[0,0]
print("True positives are", TP5)
FN5=cm5[0,1]
print("False negatives are", FN5)
FP5=cm5[1,0]
print("False positives are", FP5)
TN5=cm5[1,1]
print("True negatives are", TN5)
TPR5=TP5/(TP5 + FN5)
print("True positive rate is", np.round(TPR5*100,2))
TNR5=TN5/(TN5 + FP5)
print("True negative rate is", np.round(TNR5*100,2))
plt.figure(figsize=(10,7))
p = sns.heatmap(cm5, annot=True, cmap="Reds", fmt='g')
plt.title('Confusion matrix for Decision Tree')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[104]:


dt.fit(X1_train,y_train)
dt_pred_without_age=dt.predict(X1_test)

dt_score_without_age=round(accuracy_score(y_test,dt_pred_without_age),4)*100
print("dt Accuracy without age: {}%".format(dt_score_without_age))

dt.fit(X2_train,y_train)
dt_pred_without_sex=dt.predict(X2_test)

dt_score_without_sex=round(accuracy_score(y_test,dt_pred_without_sex),4)*100
print("dt Accuracy without sex: {}%".format(dt_score_without_sex))

dt.fit(X3_train,y_train)
dt_pred_without_onthyroxine=dt.predict(X3_test)

dt_score_without_onthyroxine=round(accuracy_score(y_test,dt_pred_without_onthyroxine),4)*100
print("dt Accuracy without onthyroxine: {}%".format(dt_score_without_onthyroxine))


dt.fit(X4_train,y_train)
dt_pred_without_queryonthyroxine=dt.predict(X4_test)

dt_score_without_queryonthyroxine=round(accuracy_score(y_test,dt_pred_without_queryonthyroxine),4)*100
print("dt Accuracy without query on thyroxine: {}%".format(dt_score_without_queryonthyroxine))

dt.fit(X5_train,y_train)
dt_pred_without_medication=dt.predict(X5_test)

dt_score_without_medication=round(accuracy_score(y_test,dt_pred_without_medication),4)*100
print("dt Accuracy without medication: {}%".format(dt_score_without_medication))


dt.fit(X6_train,y_train)
dt_pred_without_sick=dt.predict(X6_test)

dt_score_without_sick=round(accuracy_score(y_test,dt_pred_without_sick),4)*100
print("dt Accuracy without sick: {}%".format(dt_score_without_sick))

dt.fit(X7_train,y_train)
dt_pred_without_pregnant=dt.predict(X7_test)

dt_score_without_pregnant=round(accuracy_score(y_test,dt_pred_without_pregnant),4)*100
print("dt Accuracy without pregnant: {}%".format(dt_score_without_pregnant))

 
dt.fit(X8_train,y_train)
dt_pred_without_surgery=dt.predict(X8_test)

dt_score_without_surgery=round(accuracy_score(y_test,dt_pred_without_surgery),4)*100
print("dt Accuracy without surgery: {}%".format(dt_score_without_surgery))

dt.fit(X9_train,y_train)
dt_pred_without_I131treatment=dt.predict(X9_test)

dt_score_without_I131treatment=round(accuracy_score(y_test,dt_pred_without_I131treatment),4)*100
print("dt Accuracy without I131 treatment: {}%".format(dt_score_without_I131treatment))

dt.fit(X10_train,y_train)
dt_pred_without_queryhypothyroid=dt.predict(X10_test)

dt_score_without_queryhypothyroid=round(accuracy_score(y_test,dt_pred_without_queryhypothyroid),4)*100
print("dt Accuracy without query hypothyroid: {}%".format(dt_score_without_queryhypothyroid))


dt.fit(X11_train,y_train)
dt_pred_without_queryhyperthyroid=dt.predict(X11_test)

dt_score_without_queryhyperthyroid=round(accuracy_score(y_test,dt_pred_without_queryhyperthyroid),4)*100
print("dt Accuracy without query hyperthyroid: {}%".format(dt_score_without_queryhyperthyroid))

dt.fit(X12_train,y_train)
dt_pred_without_lithium=dt.predict(X12_test)

dt_score_without_lithium=round(accuracy_score(y_test,dt_pred_without_lithium),4)*100
print("dt Accuracy without lithium: {}%".format(dt_score_without_lithium))

dt.fit(X13_train,y_train)
dt_pred_without_goitre=dt.predict(X13_test)

dt_score_without_goitre=round(accuracy_score(y_test,dt_pred_without_goitre),4)*100
print("dt Accuracy without goitre: {}%".format(dt_score_without_goitre))

dt.fit(X14_train,y_train)
dt_pred_without_tumor=dt.predict(X14_test)

dt_score_without_tumor=round(accuracy_score(y_test,dt_pred_without_tumor),4)*100
print("dt Accuracy without tumor: {}%".format(dt_score_without_tumor))

dt.fit(X15_train,y_train)
dt_pred_without_hypopituitary=dt.predict(X15_test)

dt_score_without_hypopituitary=round(accuracy_score(y_test,dt_pred_without_hypopituitary),4)*100
print("dt Accuracy without hypopituitary: {}%".format(dt_score_without_hypopituitary))

dt.fit(X16_train,y_train)
dt_pred_without_psych=dt.predict(X16_test)

dt_score_without_psych=round(accuracy_score(y_test,dt_pred_without_psych),4)*100
print("dt Accuracy without psych: {}%".format(dt_score_without_psych))

dt.fit(X17_train,y_train)
dt_pred_without_TSHmeasured=dt.predict(X17_test)

dt_score_without_TSHmeasured=round(accuracy_score(y_test,dt_pred_without_TSHmeasured),4)*100
print("dt Accuracy without TSH measured: {}%".format(dt_score_without_TSHmeasured))

dt.fit(X18_train,y_train)
dt_pred_without_TSH=dt.predict(X18_test)

dt_score_without_TSH=round(accuracy_score(y_test,dt_pred_without_TSH),4)*100
print("dt Accuracy without TSH: {}%".format(dt_score_without_TSH))

dt.fit(X19_train,y_train)
dt_pred_without_T3measured=dt.predict(X19_test)

dt_score_without_T3measured=round(accuracy_score(y_test,dt_pred_without_T3measured),4)*100
print("dt Accuracy without T3 measured: {}%".format(dt_score_without_T3measured))

dt.fit(X20_train,y_train)
dt_pred_without_T3=dt.predict(X20_test)

dt_score_without_T3=round(accuracy_score(y_test,dt_pred_without_T3),4)*100
print("dt Accuracy without T3: {}%".format(dt_score_without_T3))

dt.fit(X21_train,y_train)
dt_pred_without_TT4measured=dt.predict(X21_test)

dt_score_without_TT4measured=round(accuracy_score(y_test,dt_pred_without_TT4measured),4)*100
print("dt Accuracy without TT4 measured: {}%".format(dt_score_without_TT4measured))

dt.fit(X22_train,y_train)
dt_pred_without_TT4=dt.predict(X22_test)

dt_score_without_TT4=round(accuracy_score(y_test,dt_pred_without_TT4),4)*100
print("dt Accuracy without TT4: {}%".format(dt_score_without_TT4))

dt.fit(X23_train,y_train)
dt_pred_without_T4Umeasured=dt.predict(X23_test)

dt_score_without_T4Umeasured=round(accuracy_score(y_test,dt_pred_without_T4Umeasured),4)*100
print("dt Accuracy without T4U measured: {}%".format(dt_score_without_T4Umeasured))

dt.fit(X24_train,y_train)
dt_pred_without_T4U=dt.predict(X24_test)

dt_score_without_T4U=round(accuracy_score(y_test,dt_pred_without_T4U),4)*100
print("dt Accuracy without T4U: {}%".format(dt_score_without_T4U))

dt.fit(X25_train,y_train)
dt_pred_without_FTImeasured=dt.predict(X25_test)

dt_score_without_FTImeasured=round(accuracy_score(y_test,dt_pred_without_FTImeasured),4)*100
print("dt Accuracy without FTI measured: {}%".format(dt_score_without_FTImeasured))

dt.fit(X26_train,y_train)
dt_pred_without_FTI=dt.predict(X26_test)

dt_score_without_FTI=round(accuracy_score(y_test,dt_pred_without_FTI),4)*100
print("dt Accuracy without FTI: {}%".format(dt_score_without_FTI))

dt.fit(X27_train,y_train)
dt_pred_without_TBGmeasured=dt.predict(X27_test)

dt_score_without_TBGmeasured=round(accuracy_score(y_test,dt_pred_without_TBGmeasured),4)*100
print("dt Accuracy without TBG measured: {}%".format(dt_score_without_TBGmeasured))


# In[121]:


from sklearn.svm import SVC
svc = SVC(kernel="linear")
svc.fit(X_train,np.ravel(y_train))
y_pred6=svc.predict(X_test)
score7 = round(accuracy_score(y_test, y_pred6),4)*100
print("Accuracy for Support Vector Classifier: {}%".format(score7))


# In[122]:


svcb = SVC(kernel="rbf")
svcb.fit(X_train,np.ravel(y_train))
y_pred6b=svcb.predict(X_test)
score7b = round(accuracy_score(y_test, y_pred6b),4)*100
print("Accuracy for Support Vector Classifier: {}%".format(score7b))


# In[123]:


svcc = SVC(kernel="poly")
svcc.fit(X_train,np.ravel(y_train))
y_pred6c=svcc.predict(X_test)
score7c = round(accuracy_score(y_test, y_pred6c),4)*100
print("Accuracy for Support Vector Classifier: {}%".format(score7c))


# In[125]:


#Linear kernel
cm6=confusion_matrix(y_test,y_pred6)
print(cm6)
TP6=cm6[0,0]
print("True positives are", TP6)
FN6=cm6[0,1]
print("False negatives are", FN6)
FP6=cm6[1,0]
print("False positives are", FP6)
TN6=cm6[1,1]
print("True negatives are", TN6)
TPR6=TP6/(TP6 + FN6)
print("True positive rate is", np.round(TPR6*100,2))
TNR6=TN6/(TN6 + FP6)
print("True negative rate is", np.round(TNR6*100,2))
plt.figure(figsize=(10,7))
p = sns.heatmap(cm6, annot=True, cmap="Reds", fmt='g')
plt.title('Confusion matrix for Support Vector Classifier')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# In[127]:


svc.fit(X1_train,y_train)
svc_pred_without_age=svc.predict(X1_test)

svc_score_without_age=round(accuracy_score(y_test,svc_pred_without_age),4)*100
print("svc Accuracy without age: {}%".format(svc_score_without_age))

svc.fit(X2_train,y_train)
svc_pred_without_sex=svc.predict(X2_test)

svc_score_without_sex=round(accuracy_score(y_test,svc_pred_without_sex),4)*100
print("svc Accuracy without sex: {}%".format(svc_score_without_sex))

svc.fit(X3_train,y_train)
svc_pred_without_onthyroxine=svc.predict(X3_test)

svc_score_without_onthyroxine=round(accuracy_score(y_test,svc_pred_without_onthyroxine),4)*100
print("svc Accuracy without onthyroxine: {}%".format(svc_score_without_onthyroxine))


svc.fit(X4_train,y_train)
svc_pred_without_queryonthyroxine=svc.predict(X4_test)

svc_score_without_queryonthyroxine=round(accuracy_score(y_test,svc_pred_without_queryonthyroxine),4)*100
print("svc Accuracy without query on thyroxine: {}%".format(svc_score_without_queryonthyroxine))

svc.fit(X5_train,y_train)
svc_pred_without_medication=svc.predict(X5_test)

svc_score_without_medication=round(accuracy_score(y_test,svc_pred_without_medication),4)*100
print("svc Accuracy without medication: {}%".format(svc_score_without_medication))


svc.fit(X6_train,y_train)
svc_pred_without_sick=svc.predict(X6_test)

svc_score_without_sick=round(accuracy_score(y_test,svc_pred_without_sick),4)*100
print("svc Accuracy without sick: {}%".format(svc_score_without_sick))

svc.fit(X7_train,y_train)
svc_pred_without_pregnant=svc.predict(X7_test)

svc_score_without_pregnant=round(accuracy_score(y_test,svc_pred_without_pregnant),4)*100
print("svc Accuracy without pregnant: {}%".format(svc_score_without_pregnant))

 
svc.fit(X8_train,y_train)
svc_pred_without_surgery=svc.predict(X8_test)

svc_score_without_surgery=round(accuracy_score(y_test,svc_pred_without_surgery),4)*100
print("svc Accuracy without surgery: {}%".format(svc_score_without_surgery))

svc.fit(X9_train,y_train)
svc_pred_without_I131treatment=svc.predict(X9_test)

svc_score_without_I131treatment=round(accuracy_score(y_test,svc_pred_without_I131treatment),4)*100
print("svc Accuracy without I131 treatment: {}%".format(svc_score_without_I131treatment))

svc.fit(X10_train,y_train)
svc_pred_without_queryhypothyroid=svc.predict(X10_test)

svc_score_without_queryhypothyroid=round(accuracy_score(y_test,svc_pred_without_queryhypothyroid),4)*100
print("svc Accuracy without query hypothyroid: {}%".format(svc_score_without_queryhypothyroid))


svc.fit(X11_train,y_train)
svc_pred_without_queryhyperthyroid=svc.predict(X11_test)

svc_score_without_queryhyperthyroid=round(accuracy_score(y_test,svc_pred_without_queryhyperthyroid),4)*100
print("svc Accuracy without query hyperthyroid: {}%".format(svc_score_without_queryhyperthyroid))

svc.fit(X12_train,y_train)
svc_pred_without_lithium=svc.predict(X12_test)

svc_score_without_lithium=round(accuracy_score(y_test,svc_pred_without_lithium),4)*100
print("svc Accuracy without lithium: {}%".format(svc_score_without_lithium))

svc.fit(X13_train,y_train)
svc_pred_without_goitre=svc.predict(X13_test)

svc_score_without_goitre=round(accuracy_score(y_test,svc_pred_without_goitre),4)*100
print("svc Accuracy without goitre: {}%".format(svc_score_without_goitre))

svc.fit(X14_train,y_train)
svc_pred_without_tumor=svc.predict(X14_test)

svc_score_without_tumor=round(accuracy_score(y_test,svc_pred_without_tumor),4)*100
print("svc Accuracy without tumor: {}%".format(svc_score_without_tumor))

svc.fit(X15_train,y_train)
svc_pred_without_hypopituitary=svc.predict(X15_test)

svc_score_without_hypopituitary=round(accuracy_score(y_test,svc_pred_without_hypopituitary),4)*100
print("svc Accuracy without hypopituitary: {}%".format(svc_score_without_hypopituitary))

svc.fit(X16_train,y_train)
svc_pred_without_psych=svc.predict(X16_test)

svc_score_without_psych=round(accuracy_score(y_test,svc_pred_without_psych),4)*100
print("svc Accuracy without psych: {}%".format(svc_score_without_psych))

svc.fit(X17_train,y_train)
svc_pred_without_TSHmeasured=svc.predict(X17_test)

svc_score_without_TSHmeasured=round(accuracy_score(y_test,svc_pred_without_TSHmeasured),4)*100
print("svc Accuracy without TSH measured: {}%".format(svc_score_without_TSHmeasured))

svc.fit(X18_train,y_train)
svc_pred_without_TSH=svc.predict(X18_test)

svc_score_without_TSH=round(accuracy_score(y_test,svc_pred_without_TSH),4)*100
print("svc Accuracy without TSH: {}%".format(svc_score_without_TSH))

svc.fit(X19_train,y_train)
svc_pred_without_T3measured=svc.predict(X19_test)

svc_score_without_T3measured=round(accuracy_score(y_test,svc_pred_without_T3measured),4)*100
print("svc Accuracy without T3 measured: {}%".format(svc_score_without_T3measured))

svc.fit(X20_train,y_train)
svc_pred_without_T3=svc.predict(X20_test)

svc_score_without_T3=round(accuracy_score(y_test,svc_pred_without_T3),4)*100
print("svc Accuracy without T3: {}%".format(svc_score_without_T3))

svc.fit(X21_train,y_train)
svc_pred_without_TT4measured=svc.predict(X21_test)

svc_score_without_TT4measured=round(accuracy_score(y_test,svc_pred_without_TT4measured),4)*100
print("svc Accuracy without TT4 measured: {}%".format(svc_score_without_TT4measured))

svc.fit(X22_train,y_train)
svc_pred_without_TT4=svc.predict(X22_test)

svc_score_without_TT4=round(accuracy_score(y_test,svc_pred_without_TT4),4)*100
print("svc Accuracy without TT4: {}%".format(svc_score_without_TT4))

svc.fit(X23_train,y_train)
svc_pred_without_T4Umeasured=svc.predict(X23_test)

svc_score_without_T4Umeasured=round(accuracy_score(y_test,svc_pred_without_T4Umeasured),4)*100
print("svc Accuracy without T4U measured: {}%".format(svc_score_without_T4Umeasured))

svc.fit(X24_train,y_train)
svc_pred_without_T4U=svc.predict(X24_test)

svc_score_without_T4U=round(accuracy_score(y_test,svc_pred_without_T4U),4)*100
print("svc Accuracy without T4U: {}%".format(svc_score_without_T4U))

svc.fit(X25_train,y_train)
svc_pred_without_FTImeasured=svc.predict(X25_test)

svc_score_without_FTImeasured=round(accuracy_score(y_test,svc_pred_without_FTImeasured),4)*100
print("svc Accuracy without FTI measured: {}%".format(svc_score_without_FTImeasured))

svc.fit(X26_train,y_train)
svc_pred_without_FTI=svc.predict(X26_test)

svc_score_without_FTI=round(accuracy_score(y_test,svc_pred_without_FTI),4)*100
print("svc Accuracy without FTI: {}%".format(svc_score_without_FTI))

svc.fit(X27_train,y_train)
svc_pred_without_TBGmeasured=svc.predict(X27_test)

svc_score_without_TBGmeasured=round(accuracy_score(y_test,svc_pred_without_TBGmeasured),4)*100
print("svc Accuracy without TBG measured: {}%".format(svc_score_without_TBGmeasured))


# In[128]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
models = ['Naive Bayes','RF','KNN','LoR','SVC','DT']
accuracy = [90.56,93.11,94.06,95.92,96.98,99.72]
ax.bar(models, accuracy)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Algorithms')
ax.set_title('Comparison of Classifiers')
plt.show()
#Best accuracy by Decision Tree


# In[642]:


#Applying hyperparameter tuning using using Randomized search and subsequent Grid search, as well as Stratified KFold Cross Validation for best classifier (i.e. Decision Tree).


# In[646]:


from sklearn.model_selection import RepeatedStratifiedKFold
seed = np.random.seed(123)
rng = np.random.RandomState(1)
from sklearn.model_selection import RandomizedSearchCV
random_grid = {"max_depth": [i for i in range(1,100)],
              "max_features": ['auto', 'sqrt','log2'],
              "min_samples_leaf" : [j for j in range(1,11)],
              "criterion": ["gini", "entropy"],
              "splitter":["best", "random"]}
dt2=DecisionTreeClassifier(random_state=rng)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=3)
dt_after_randomizedsearch=RandomizedSearchCV(estimator=dt2,param_distributions=random_grid,n_iter=100,cv=cv)
dt_after_randomizedsearch.fit(X_train,y_train)
dt_after_randomizedsearch.best_params_
# {'splitter': 'best',
#  'min_samples_leaf': 1,
#  'max_features': 'sqrt',
#  'max_depth': 9,
#  'criterion': 'gini'}
dt2 = DecisionTreeClassifier(splitter='best',
 min_samples_leaf=1,
 max_features= 'sqrt',
 max_depth= 9,
 criterion= 'gini')
dt2.fit(X_train,np.ravel(y_train))
y_predb=dt2.predict(X_test)
scoreb = round(accuracy_score(y_test, y_predb),4)*100
print("Accuracy for Decision Tree after Randomized Search: {}%".format(scoreb))


# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {"max_depth":[i for i in range(100,200)],
              "max_features": [dt_after_randomizedsearch.best_params_['max_features']],
              "min_samples_leaf" : [11,12,13,14,15,16,17,18,19,20,dt_after_randomizedsearch.best_params_['min_samples_leaf']],
              "criterion": [dt_after_randomizedsearch.best_params_['criterion']],
              "splitter":[dt_after_randomizedsearch.best_params_['splitter']]}   
dt3=DecisionTreeClassifier(random_state=rng)
dt_after_gridsearch=GridSearchCV(estimator=dt3,param_grid=param_grid,cv=cv)
dt_after_gridsearch.fit(X_train,y_train)
dt_after_gridsearch.best_estimator_              

dt4 = DecisionTreeClassifier(splitter='best',
 min_samples_leaf=1,
 max_features= 'auto',
 max_depth= 200,
 criterion= 'entropy')
dt4.fit(X_train,np.ravel(y_train))
y_predc=dt4.predict(X_test)
scorec = round(accuracy_score(y_test, y_predc),4)*100
print("Accuracy for Decision Tree after Grid Search: {}%".format(scorec))


# In[512]:


from sklearn.metrics import precision_score
cm_dt_final=confusion_matrix(y_test,y_predc)
print(cm_dt_final)


TPc=cm_dt_final[0,0]
print("True positives are", TPc)
FNc=cm_dt_final[0,1]
print("False negatives are", FNc)
FPc=cm_dt_final[1,0]
print("False positives are", FPc)
TNc=cm_dt_final[1,1]
print("True negatives are", TNc)
TPRc=TPc/(TPc + FNc)
print("True positive rate or Recall or Selectivity is",np.round(TPRc*100,2))
TNRc=TNc/(TNc + FPc)
print("True negative rate or Specificity is",np.round(TNRc*100,2))
PPV=TPc/(TPc+FPc)
print("Positive predicted value or precision is",np.round(PPV,2))
NPV=TNc/(TNc+FNc)
print("Negative predicted value is",np.round(NPV,2))
F1score=2*TPc/(2*TPc + FPc + FNc)
print("F1 score is",np.round(F1score,2))
FNRc=FNc/(FNc + TPc)
print("False negative rate or miss rate is",np.round(FNRc,2))
FPRc=FPc/(FPc + TNc)
print("False positive rate or fallout rate is",np.round(FPRc,2))
FDRc=FPc/(FPc + TPc)
print("False discovery rate is",np.round(FDRc,2))
FORc=FNc/(FNc + TNc)
print("False omission rate is",np.round(FORc,2))


# In[517]:


from sklearn.metrics import roc_curve
def plot_roc():
    plt.plot(FPRc, TPRc, linewidth = 2) 
    plt.plot([0,1],[0,1], 'k--', linewidth = 2)
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('ROC Curve for Decision tree (after hyperparameter tuning)')
    plt.show()
FPRc, TPRc, p= roc_curve(y_test, y_predc)
plot_roc()

# In[519]:


from sklearn.metrics import roc_auc_score
auc = roc_auc_score( y_test, y_predc)
print("ROC-AUC score is",np.round(auc,2))


# In[164]:


from sklearn.cluster import KMeans
X_Kmeans = df.iloc[:, [0,19]] #age,T3
X_scaled= scaler.fit_transform(X_Kmeans)
inertia_list=[]
for k in range (1,9): #for different number of clusters
    kmeans_classifier = KMeans(n_clusters=k)
    y_kmeans = kmeans_classifier.fit_predict(X_scaled)
    inertia = kmeans_classifier.inertia_
    inertia_list.append(inertia) #computing inertia 
fig,ax = plt.subplots (1 , figsize =(6,4))
plt.plot(range(1,9),inertia_list,marker ='o',color ='black',linestyle='dashed')
plt.xlabel('Number of clusters k')
plt.ylabel('Inertia or distortion')
plt.tight_layout ()
plt.show()

#Best k(number of clusters)=4
kmeans_classifier2 =KMeans(n_clusters=4)
y_kmeans2 = kmeans_classifier2.fit_predict(X_scaled)
centroids = kmeans_classifier2.cluster_centers_
print("Cluster centroids are:", centroids)

df['Cluster']=kmeans_classifier2.labels_ #getting cluster label
Cluster=df['Cluster'].tolist()
Clustering= list(zip(df['Outcome'],Cluster)) #true label, cluster label
cluster0=[]
cluster1=[]
cluster2=[]
cluster3=[]
for i in Clustering: #separate lists for each cluster
    if i[1]==0:
        cluster0.append(i[0])
    elif i[1]==1:
        cluster1.append(i[0])
    elif i[1]==2:
        cluster2.append(i[0])
    elif i[1]==3:
        cluster3.append(i[0])
count_healthy=0
count_unhealthy=0
for i in cluster0: #healthy vs unhealthy labels in first cluster
    if i==0:
        count_healthy=count_healthy+1
    else:
        count_unhealthy=count_unhealthy+1
percent_healthy=count_healthy/(count_healthy+count_unhealthy)*100
percent_unhealthy=count_unhealthy/(count_healthy+count_unhealthy)*100
print("Percentage of healthy and unhealthy labels in first cluster is", np.round(percent_healthy,2),"%","and", np.round(percent_unhealthy,2),"%","respectively.")

count_healthy1=0
count_unhealthy1=0
for i in cluster1:
    if i==0:
        count_healthy1=count_healthy1+1
    else:
        count_unhealthy1=count_unhealthy1+1
percent_healthy1=count_healthy1/(count_healthy1+count_unhealthy1)*100
percent_unhealthy1=count_unhealthy1/(count_healthy1+count_unhealthy1)*100
print("Percentage of healthy and unhealthy labels in second cluster is", np.round(percent_healthy1,2),"%","and", np.round(percent_unhealthy1,2),"%","respectively.")

count_healthy2=0
count_unhealthy2=0
for i in cluster2:
    if i==0:
        count_healthy2=count_healthy2+1
    else:
        count_unhealthy2=count_unhealthy2+1
percent_healthy2=count_healthy2/(count_healthy2+count_unhealthy2)*100
percent_unhealthy2=count_unhealthy2/(count_healthy2+count_unhealthy2)*100
print("Percentage of healthy and unhealthy labels in third cluster is", np.round(percent_healthy2,2),"%","and", np.round(percent_unhealthy2,2),"%","respectively.")

count_healthy3=0
count_unhealthy3=0
for i in cluster3:
    if i==0:
        count_healthy3=count_healthy3+1
    else:
        count_unhealthy3=count_unhealthy3+1
percent_healthy3=count_healthy3/(count_healthy3+count_unhealthy3)*100
percent_unhealthy3=count_unhealthy3/(count_healthy3+count_unhealthy3)*100
print("Percentage of healthy and unhealthy labels in fourth cluster is",np.round(percent_healthy3,2),"%","and", np.round(percent_unhealthy3,2),"%","respectively.")


#Visualising the clusters
plt.scatter(X_scaled[y_kmeans2 == 0, 0], X_scaled[y_kmeans2 == 0, 1], s = 20, c = 'deeppink', label = 'Cluster 0')
plt.scatter(X_scaled[y_kmeans2 == 1, 0], X_scaled[y_kmeans2 == 1, 1], s = 20, c = 'lightpink', label = 'Cluster 1')
plt.scatter(X_scaled[y_kmeans2 == 2, 0], X_scaled[y_kmeans2 == 2, 1], s = 20, c = 'plum', label = 'Cluster 2')
plt.scatter(X_scaled[y_kmeans2 == 3, 0], X_scaled[y_kmeans2 == 3, 1], s = 20, c = 'darkorchid', label = 'Cluster 3')

plt.scatter(x=kmeans_classifier2.cluster_centers_[:, 0], y=kmeans_classifier2.cluster_centers_[:, 1], s=100, c='black', marker='+', label='Cluster Centers')
plt.legend()
plt.title('Clusters of patients')
plt.xlabel('Age')
plt.ylabel('T3')
plt.show()


# In[134]:


#Cluster0: Patients with lower age and lower T3
#Cluster1: Patients with comparatively higher age and lower T3
#Cluster2: Patients with higher age and slighty higher T3
#Cluster4: Patients with higher T3.

