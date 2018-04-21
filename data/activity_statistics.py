import pandas as pd
import numpy as np
import math
import sys
from signal import *
import calendar
from datetime import datetime

csv_to_read = 'tung_hist_jan_mar_weather_nolocomotion_mood'
df = pd.read_csv(csv_to_read + '.csv', index_col=0)

stats_df = df['Name'].value_counts()
stats_df.to_csv('activity_frequency.csv')
stats_df = stats_df.to_dict()
stats_df = sorted(stats_df.items(), key=lambda kv: kv[1], reverse=True)

mean_df = df.groupby('Name', as_index=False)['Mood'].mean()

new_df = pd.DataFrame()
for i in stats_df:	
	new_df = new_df.append(mean_df.loc[mean_df['Name'] == i[0]])
new_df.dropna(subset=['Mood'], inplace=True)

new_df.to_csv('average_mood.csv')

