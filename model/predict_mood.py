import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import tree
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from sklearn import preprocessing
import itertools
#import seaborn as sns
from sklearn.externals import joblib
import sys

import os
script_dir = os.path.dirname(os.path.realpath(__file__))

def df_encode(df):
	le  = preprocessing.LabelEncoder()
	ohe = preprocessing.OneHotEncoder(sparse=False)

	le.fit(df)
	np_ = le.fit_transform(df)
	np_ = np_.reshape((len(np_),1))
	ohe.fit(np_)
	np_ = ohe.fit_transform(np_)
	df  = pd.DataFrame(np_)
	return df, le, ohe

def df_transform(df, le, ohe):
	return pd.DataFrame(ohe.transform(le.transform(df).reshape(len(df),1)))

def encode_features(df, send_to_csv):
	# DROP PLACES WITHOUT CATEGORIES
	df.fillna("other", inplace=True)
	#encode features
	# name_le                                = joblib.load(script_dir + '/../model/name_le.pkl')
	# name_ohe                               = joblib.load(script_dir + '/../model/name_ohe.pk1')
	# name_df                                = df_transform(df['Name'], name_le, name_ohe)
	time_df                                = df['BeginTime'].str.split(':').str.get(0).apply(lambda x: int(x))
	week_df                                = df['WeekDay']
	duration_df                            = df['Duration'].str.split('h').str.get(0).apply(lambda x: int(x))
	category_le                            = joblib.load(script_dir + '/../model/category_le.pkl')
	category_ohe                           = joblib.load(script_dir + '/../model/category_ohe.pk1')
	category_df                            = df_transform(df['Category'], category_le, category_ohe)
	weather_le                             = joblib.load(script_dir + '/../model/weather_le.pkl')
	weather_ohe                            = joblib.load(script_dir + '/../model/weather_ohe.pk1')
	weather_df                             = df_transform(df['Weather'], weather_le, weather_ohe)
	people_df                              = df.filter(regex='People')

	# table_df= pd.concat([ time_df, week_df, duration_df, weather_df, people_df], axis=1, join='inner').dropna(how='any', axis=0)
	table_df= pd.concat([week_df, duration_df, weather_df, category_df, people_df], axis=1, join='inner').fillna("other", axis=0)
#	print table_df
	if send_to_csv:
		table_df.to_csv('encoded_'+sys.argv[1])
	return table_df



test_df         = pd.read_csv(sys.argv[1], index_col=0)
encoded_test_df = encode_features(test_df, True)
#print test_df
#print encoded_test_df

filename = script_dir + '/../model/Mood_Predictor_1to3.sav'
clf_multiclass_bot = joblib.load(filename)
filename = script_dir + '/../model/Mood_predictor_4to5.sav'
clf_multiclass_top = joblib.load(filename)


y_pred_bot = clf_multiclass_bot.predict_proba(encoded_test_df.values) #predict_proba
y_pred_top = clf_multiclass_top.predict_proba(encoded_test_df.values) #predict_proba

y_pred_bot = y_pred_bot[:,0:3]
y_pred_top = y_pred_top[:,3:5]
y_pred     = np.concatenate((y_pred_bot, y_pred_top), axis=1)

y_pred_mood = np.argmax(y_pred, axis=1)

y_pred_scaled = np.zeros_like(y_pred[0])

for i in range(len(y_pred[0])):
	y_pred_scaled[i] = y_pred[0][i] * (i + 1)

print y_pred_scaled

print np.sum(y_pred)
y_pred_scaled = float(np.sum(y_pred_scaled)) / float(np.sum(y_pred[0]))


print y_pred_scaled

pd.DataFrame({'Mood':y_pred_mood}).to_csv('predicted_mood.csv')
