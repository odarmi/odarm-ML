import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import cross_validation
from collections import defaultdict
from sklearn import preprocessing

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

def extract_features(df, send_to_csv):
	#extract features
	name_df, name_le, name_ohe             = df_encode(df['Name'])
	time_df                                = df['BeginTime'].str.split(':').str.get(0)
#	week_df                                = df['WeekDay']
	week_df, week_le, week_ohe	       = df_encode(df['WeekDay'])
	duration_df                            = df['Duration'].str.split('h').str.get(0)
	category_df, category_le, category_ohe = df_encode(df['Category'])
	weather_df, weather_le, weather_ohe    = df_encode(df['Weather']) 
	mood_df                                = df['Mood']
	
	#%matplotlib inline
	#plt.scatter(dataset[:,0], dataset[:,1], c='blue', s=10)
	#plot.show()
	
	table_df= pd.concat([name_df, time_df, week_df, duration_df, category_df, weather_df, mood_df], axis=1, join='inner').dropna(how='any', axis=0)
	#print table_df
	if send_to_csv:
		table_df.to_csv('features.csv')
	return table_df

df = pd.read_csv("tung_hist_jan_mar_weather_nolocomotion_mood.csv", index_col=0)
table_df = extract_features(df, True)

feature_df = table_df.drop('Mood', axis=1)
label_df   = table_df['Mood']
#print feature_df
#print label_df


clf = tree.DecisionTreeClassifier()
scores = cross_validation.cross_val_score(clf, feature_df, label_df, cv=10)
print scores
print scores.mean() 
print scores.std()*2


#clf = clf.fit(feature_df, label_df)
#print feature_df.iloc[0:1]
#print clf.predict(feature_df.iloc[0:1])
#print clf.predict(feature_df[0])

#pd.concat([df_transform(['Home'], name_le, name_ohe),pd.DataFrame([9]),pd.DataFrame([3]),pd.DataFrame([4]), df_transform(['Apartment Building'], category_le, category_ohe), df_transform([' clear-day'], weather_le, weather_ohe)]))



