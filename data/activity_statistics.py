import pandas as pd
import numpy as np
import math
import sys
from signal import *
import calendar
from datetime import datetime


csv_to_read = 'tung_hist_jan_mar_weather_nolocomotion_mood'
df = pd.read_csv(csv_to_read + '.csv', index_col=0)

print df['Name'].value_counts()

#while True:
#	try:
#		activity = input('')
#		print  df.groupby(	
#	except (TypeError, SyntaxError, ValueError, NameError) as e:
#		print "Enter valid activity"
#		continue
#print "\n\n"
