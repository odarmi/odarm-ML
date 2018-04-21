import pandas as pd
import numpy as np

df = pd.read_csv('tung_hist_jan_mar_weather.csv', index_col=0)
df.dropna(subset=['Address'], inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv('tung_hist_jan_mar_weather_nolocomotion.csv')
