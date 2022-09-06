import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('StudentsPerformance.csv')

df['total score'] = df['math score'] + df['reading score'] + df['writing score']
df['average score'] = round(df['total score'] / 3, 1)

print(df['lunch'].value_counts())
print('-' * 25)
print(df['parental level of education'].value_counts())
print('-' * 25)
print(df['gender'].value_counts())
print('-' * 25)
print(df['race/ethnicity'].value_counts())
print('-' * 25)
print(df['test preparation course'].value_counts())



sns.barplot(data=df, y='average score', x='race/ethnicity')






df.groupby('race/ethnicity')['average score'].mean()


fig, ax = plt.subplots(1,2, figsize=(12,4))
xys = [20,100]

sns.scatterplot(ax=ax[0], data=df, x='reading score', y='writing score', hue='gender')
sns.lineplot(ax=ax[0], x=xys, y=xys, color='red', linestyle='--')

sns.scatterplot(ax=ax[1], data=df, x='reading score', y='math score', hue='gender')
sns.lineplot(ax=ax[1], x=xys, y=xys, color='red', linestyle='--')

plt.show()

gender_map = {'male': 1, 'female': 0}
df['gender'] = df['gender'].map(gender_map)
df['gender'] = df['gender'].astype('int64')

test_map = {'completed': 1, 'none': 0}
df['test preparation course'] = df['test preparation course'].map(test_map)
df['test preparation course'] = df['test preparation course'].astype('int64')


lunch_map = {'standard': 1, 'free/reduced': 0}
df['lunch'] = df['lunch'].map(lunch_map)
df['lunch'] = df['lunch'].astype('int64')

edu_map =  {'some high school': 0, 'high school': 1, 'some college' : 2, 'college' : 3, "associate's degree" : 4, "bachelor's degree" : 5, "master's degree" : 6}
df['parental level of education'] = df['parental level of education'].map(edu_map)
df['parental level of education'] = df['parental level of education'].astype('int64')


raceethnicity_map = {"group A" : 1,"group B": 2,"group C" : 3,"group D" : 4, "group E" : 5}
df['race/ethnicity'] = df['race/ethnicity'].map(raceethnicity_map)
df['race/ethnicity'] = df['race/ethnicity'].astype('int64')



X = df[['reading score', 'math score', 'average score', 'total score','race/ethnicity', 'lunch', 'test preparation course']]
y = df['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)




model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score_logreg = accuracy_score(y_test, y_pred)

results = X_test.copy()
results['gender_true'] = y_test
results['gender_predicted'] = y_pred

print(len(results[results.gender_true != results.gender_predicted]))