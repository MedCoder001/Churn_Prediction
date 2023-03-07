import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score
from IPython.display import display
from sklearn.feature_extraction import DictVectorizer


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'

import subprocess

subprocess.run(["wget", data, "-O", "data_for_churn.csv"])


df = pd.read_csv("data_for_churn.csv")
df.head()


df.columns = df.columns.str.lower().str.replace(' ', '_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


 
df.head().T


tc=pd.to_numeric(df.totalcharges, errors='coerce')


df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)


df[tc.isnull()][['customerid', 'totalcharges']]


df.churn.head()


df.churn = (df.churn == 'yes').astype(int)


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)


len(df_train), len(df_test), len(df_val)


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values


del df_train['churn']
del df_val['churn']
del df_test['churn']


y_test


df_full_train = df_full_train.reset_index(drop=True)


df_full_train.churn.value_counts(normalize=True)


global_churn_rate = df_full_train.churn.mean().round(2)
global_churn_rate


numerical = ['tenure', 'monthlycharges', 'totalcharges']


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice', 
               'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 
               'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling','paymentmethod']


df_full_train[categorical].nunique()


df_full_train.head()


churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female


churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male


global_churn_rate = df_full_train.churn.mean()
global_churn_rate


df_full_train.partner.value_counts()


churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_partner


churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_no_partner


global_churn_rate - churn_partner


churn_partner/global_churn_rate


churn_no_partner/global_churn_rate


for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean'] / global_churn_rate
    display(df_group)
    print()
    print()


mutual_info_score(df_full_train.churn, df_full_train.contract)


mutual_info_score(df_full_train.churn, df_full_train.gender)


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)



mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending=False)


df_full_train[numerical].corrwith(df_full_train.churn)


df_full_train[df_full_train.tenure <= 2].churn.mean()


df_full_train[df_full_train.tenure > 2].churn.mean()


dv = DictVectorizer(sparse=False)

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_train, y_train)


X_train.shape


val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()


model.intercept_[0]


model.coef_[0].round(3)


df_val[churn_decision].customerid.head()


y_val


churn_decision.astype(int)


df_pred = pd.DataFrame()
df_pred['customerid'] = df_val.customerid
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
df_pred['probability'] = y_pred


df_pred.head()


df_pred['correct'] = df_pred.prediction == df_pred.actual


df_pred


df_pred.correct.mean()


dv.get_feature_names_out()


model.coef_[0].round(3)


model.intercept_[0]


dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))


# Using The Model


dicts_train_full = df_full_train[categorical + numerical].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_train_full)
y_full_train = df_full_train.churn.values

model = LogisticRegression(solver='liblinear', random_state=1)
model.fit(X_full_train, y_full_train)


dicts_test = df_test[categorical + numerical].to_dict(orient='records')

X_test = dv.transform(dicts_test)
y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred >= 0.5)


(y_test == churn_decision).mean()


1145/1409


accuracy_score(y_test, y_pred >= 0.5)


thresholds = np.linspace(0, 1, 21)
scores = []
for t in thresholds:
    score = accuracy_score(y_test, y_pred >= t)
    print("%.2f %.3f" % (t, score))
    scores.append(score)


scores


plt.plot(thresholds, scores)


from collections import Counter


Counter(y_pred >= 1.0)


1 - y_val.mean()

 
# Confusion Table


actual_postive = (y_val == 1)
actual_negative = (y_val == 0)


t = 0.5
predicted_positive = (y_pred >= t)
predicted_negative = (y_pred < t)


tp = (predicted_positive & actual_postive).sum()
tn = (predicted_negative & actual_negative).sum()

fp = (predicted_positive & actual_negative).sum()
fn = (predicted_negative & actual_postive).sum()


tp, tn, fp, fn


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix


(confusion_matrix/confusion_matrix.sum()).round(2)


from sklearn.metrics import confusion_matrix


confusion_matrix(y_val, y_pred >= 0.5)

# Precision and Recall


precision = tp / (tp + fp) 


precision


recall = tp / (tp + fn)
recall


from sklearn.metrics import precision_score, recall_score


precision_score(y_val, y_pred >= 0.5)


recall_score(y_val, y_pred >= 0.5)

 
# Roc curve

 
# TPR and FPR


tpr = tp/(tp+fn)
tpr


fpr = fp/(fp+tn)
fpr


scores = []
thresholds = np.linspace(0, 1, 101)
for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)

    predicted_positive = (y_pred >= t)
    predicted_negative = (y_pred < t)

    tp = (predicted_positive & actual_positive).sum()
    tn = (predicted_negative & actual_negative).sum()
    
    fp = (predicted_positive & actual_negative).sum()
    fn = (predicted_negative & actual_positive).sum()
    scores.append((t, tp, tn, fp, fn))


columns = ['threshold', 'tp', 'tn', 'fp', 'fn']
df_scores = pd.DataFrame(scores, columns = columns)


df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)


df_scores[::10]


plt.plot(df_scores.threshold, df_scores.tpr, label='TPR')
plt.plot(df_scores.threshold, df_scores.fpr, label='FPR')
plt.legend()

 
# Random Model


np.random.seed(1)


y_rand = np.random.uniform(0, 1, size=len(y_val))
y_rand.round(3)


((y_rand >= 0.5) == y_val).mean()


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []
    thresholds = np.linspace(0, 1, 101)
    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)
        predicted_positive = (y_pred >= t)
        predicted_negative = (y_pred < t)

        tp = (predicted_positive & actual_positive).sum()
        tn = (predicted_negative & actual_negative).sum()
        fp = (predicted_positive & actual_negative).sum()
        fn = (predicted_negative & actual_positive).sum()
        scores.append((t, tp, tn, fp, fn))

    columns = ['threshold', 'tp', 'tn', 'fp', 'fn']
    df_scores = pd.DataFrame(scores, columns = columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    return df_scores


df_rand = tpr_fpr_dataframe(y_val, y_rand)


df_rand[::10]


plt.plot(df_rand.threshold, df_rand.tpr, label='TPR')
plt.plot(df_rand.threshold, df_rand.fpr, label='FPR')
plt.legend()


# Ideal Model


num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()


num_neg, num_pos


y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal


y_ideal_pred = np.linspace(0, 1, len(y_val))


(1 - y_val).mean()


((y_ideal_pred >= 0.726) == y_ideal).mean()


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)


df_ideal[::10]


plt.plot(df_ideal.threshold, df_ideal.tpr, label='TPR')
plt.plot(df_ideal.threshold, df_ideal.fpr, label='FPR')
plt.legend()

 
# Putting Everything Together


plt.plot(df_scores.threshold, df_scores.tpr, label='TPR')
plt.plot(df_scores.threshold, df_scores.fpr, label='FPR')

plt.plot(df_rand.threshold, df_rand.tpr, label='TPR')
plt.plot(df_rand.threshold, df_rand.fpr, label='FPR')

plt.plot(df_ideal.threshold, df_ideal.tpr, label='TPR')
plt.plot(df_ideal.threshold, df_ideal.fpr, label='FPR')

plt.legend()


plt.plot(df_scores.fpr, df_scores.tpr, label='model')
plt.plot(df_rand.fpr, df_rand.tpr, label='random')
plt.plot(df_ideal.fpr, df_ideal.tpr, label='ideal')
plt.legend()
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.figure(figsize=(6, 6))


from sklearn.metrics import roc_curve


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


plt.plot(fpr, tpr, label='model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.legend()

 
# ROC AUC


from sklearn.metrics import auc


auc(fpr, tpr)


auc(df_scores.fpr, df_scores.tpr)


from sklearn.metrics import roc_auc_score


roc_auc_score(y_val, y_pred)


neg = (y_val == 0)
pos = (y_val == 1)


import random


n = 100000
success = 0
for i in range(n):
    pos_ind = random.randint(0, len(pos)-1)
    neg_ind = random.randint(0, len(neg)-1)
    if pos[pos_ind] > neg[neg_ind]:
        success += 1
        
success/n        

 
# Cross Validation


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


dv, model = train(df_train, y_train, C=0.0001)


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


y_pred = predict(df_val, dv, model)


from sklearn.model_selection import KFold


kfold = KFold(n_splits=10, shuffle=True, random_state=1)


subprocess.run(["pip", "install", "tqdm"])


from tqdm.auto import tqdm

C = 1.0
n_splits = 5
print(f'Doing validation with C={C}')
for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    scores = []  # reset the scores list for each value of C

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    fold = 0
    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
    
        y_train = df_train.churn.values
        y_val = df_val.churn.values
    
        dv, model = train(df_train, y_train)
        y_pred = predict(df_val, dv, model)
    
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
        print(f'auc on fold {fold} is {auc}')
        fold += 1

    print('validation results:')
    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


print('Training the final model')
#Training the final model

dv, model = train(df_full_train, df_full_train.churn.values, C=0.1)
y_pred = predict(df_test, dv, model)
    
auc = roc_auc_score(y_test, y_pred)

print(f'auc {auc}')

 
# Saving the Model


import pickle


output_file = f'model_C={C}.bin'
output_file


f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


#with open(output_file, 'wb') as f_in:
   #dv, model = pickle.load(f_in)

print(f'The model is saved to {output_file}')
