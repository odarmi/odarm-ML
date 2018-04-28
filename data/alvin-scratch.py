
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[1]:


df = pd.read_csv('tung_hist_jan_mar_weather_nolocomotion_people_mood.csv')
df = df.drop(['Unnamed: 0', 'Distance'], axis=1)


# In[2]:


df.head()


# In[ ]:


def df_encode(df):
    return pd.get_dummies(df)


df['BeginDate'] = pd.to_datetime(df['BeginDate'])
# DROP PLACES ONLY SEEN LESS THAN x TIMES
df = df.groupby('Name').filter(lambda x: len(x) >= 3)
# DROP PLACES WITHOUT CATEGORIES
df = df.dropna()

#extract features
name_df           = df_encode(df['Name'])
time_df                                = df['BeginTime'].str.split(':').str.get(0)
week_df                                = df['WeekDay']
week_df               = df_encode(df['WeekDay'])
duration_df                            = df['Duration'].str.split('h').str.get(0)
category_df = df_encode(df['Category'])
weather_df    = df_encode(df['Weather']) 
people_df                              = df.filter(regex='People') 
mood_df                                = df['Mood']

table_df= pd.concat([name_df, time_df, week_df, duration_df, category_df, weather_df, people_df, mood_df], axis=1, join='inner').dropna(how='any', axis=0)


# In[ ]:


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(table_df.corr(), cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


def show_dists_for_person(df, person, ax):
    not_present = df[df[person] == 0]['Mood']
    present = df[df[person] == 1]['Mood']
    sns.distplot(present, rug=True, label="{0} present".format(person), ax=ax)
    sns.distplot(not_present, rug=True, label="{0} not present".format(person), ax=ax)
    ax.legend(loc='upper left')
    return not_present, present


# In[ ]:


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(10, 10))

ns, s = show_dists_for_person(table_df, 'People-Stacy Lan', ax)

