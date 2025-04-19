# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Prabhakaran P
RegisterNumber:  212224040236
*/
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

## TOP 5 ELEMENTS

![311938296-0a5cda11-f165-4e1b-86ec-5f16a2f1ee09](https://github.com/user-attachments/assets/9f6aae48-fe52-41bc-afec-810f59042561)


![311938319-01a8cd00-a0ac-49e9-bdc5-116dc5c20f3d](https://github.com/user-attachments/assets/d0bd4806-afa3-4583-9d09-887fbdf03a23)

![311938372-877b2b6f-3436-47e1-9833-e0f9ad9aa560](https://github.com/user-attachments/assets/a73af079-0442-44c4-8d75-00b9adbd31e7)

![311938372-877b2b6f-3436-47e1-9833-e0f9ad9aa560](https://github.com/user-attachments/assets/a7cbcc1a-ad2d-439d-b6b7-6abf25967463)

## Data Duplicate:

```
0
```

## Print Data:

![311938555-8ff146fd-7c1b-4323-8bba-b9fedaee4ab1](https://github.com/user-attachments/assets/b2b4aa17-9ca5-41d9-8576-cf3d1a9219d2)

## Data-Status:

![311938584-8e99861d-c573-4987-9024-596a68482332](https://github.com/user-attachments/assets/bbe84b67-c8a2-496b-8d99-9aa2d2b5129f)

## y_prediction array:

![311938611-a0cf6d5c-79d4-485e-84ea-e6601f115379](https://github.com/user-attachments/assets/82350379-e42a-473d-990b-d069c902795b)

## Confusion array:

![311938646-58525a0d-f694-4ddf-ac84-5b596381c5ef](https://github.com/user-attachments/assets/310eff58-97f1-4afe-a287-0ba7780cbb82)

## Accuracy Value:

![311938666-4f64023f-c3c2-45d2-afca-b68b780f5279](https://github.com/user-attachments/assets/b4569548-b754-43c8-bcc9-724950eb4ff2)


## Classification Report:
![311938696-9a18c485-4dcb-4116-b6d5-f8fdab5de668](https://github.com/user-attachments/assets/eb495f8f-6a01-4fa7-ae1a-7c4b971632a2)

## Prediction of LR:

![311938726-e0aeefa2-a16d-40cd-b5ab-b1b1a22044f1](https://github.com/user-attachments/assets/4952f825-db8e-4238-b375-edb0d6f9a4a6)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
