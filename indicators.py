import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,PrecisionRecallDisplay,accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

data= pd.read_csv("NHANES.csv",sep=',')
data

df = pd.DataFrame(data)
df

df.dtypes

df = df.drop(['SEQN'],axis=1)
df

df = df.rename(columns={'RIDAGEYR':'Age'})
df

df['age_group'] = np.where(df['Age']>=65,"Senior","Non-Senior")
df['age_group']

grp_cnt = df['age_group'].value_counts()
grp_cnt

fig1 = px.pie(names = grp_cnt.index,values=grp_cnt)
fig1.show()

le = LabelEncoder()
df['age_group'] = le.fit_transform(df['age_group'])
df

sns.pairplot(df)
plt.show()


x = df.drop(['age_group'],axis=1)
y = df.age_group

x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = LogisticRegression()
lr.fit(x_train,y_train.ravel())
y_pred = lr.predict(x_test)
acc_lr = accuracy_score(y_test,y_pred)
print(acc_lr)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

rf = RandomForestClassifier()
rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)
acc_rf = accuracy_score(y_test,rf_pred)
print(acc_rf)
print(confusion_matrix(y_test,rf_pred))
print(classification_report(y_test,rf_pred))

x.columns
len(df.columns)

PrecisionRecallDisplay.from_estimator(rf,x_test,y_test)
plt.show()

import pickle
pickle.dump(rf,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))