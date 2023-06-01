# Sparktask1
 PREDICTIONS USING SUPERVISED MACHINE LEARNING MODEL
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

     
IMPORTING DATA


df=pd.read_csv('/content/student_scores - student_scores.csv')
     

df.head()
     
Hours	Scores
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
CHEAKING NULL VALUES


df.isna().sum()
     
Hours     0
Scores    0
dtype: int64


     
CHEAKING CORRELATION


corr = df.corr(method = 'pearson')
corr
     
Hours	Scores
Hours	1.000000	0.976191
Scores	0.976191	1.000000

plt.scatter(df['Hours'],df['Scores'])
plt.title("HOURS VS SCORES")
plt.xlabel("HOURS")
plt.ylabel("SCORES")
     
Text(0, 0.5, 'SCORES')

DEFINING DEPENDENT AND INDEPENDENT VARIABLES AND SPLITING DATA INTO TRAINING AND TESTING SET


X=df['Hours'].values.reshape(-1,1)
Y=df['Scores'].values.reshape(-1,1)
     

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
     
FITTING A LINEAR REGRESSION MODEL


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
     
LinearRegression()
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
PREDICTION AND EVALUATION


y_pred = regressor.predict(x_test)
print(y_pred)
     
[[17.05366541]
 [33.69422878]
 [74.80620886]
 [26.8422321 ]
 [60.12335883]
 [39.56736879]
 [20.96909209]
 [78.72163554]]

import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2)) 
print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))
     
Mean absolute error = 4.42
Mean squared error = 22.97
Median absolute error = 3.86
Explain variance score = 0.96
R2 score = 0.96
PREDICTING THE SCORE OF A STUDENT WHO IS STUDYING 9.25 HRS PER DAY


Prediction=[[9.25]]
regressor.predict(Prediction)
     
array([[92.91505723]])
