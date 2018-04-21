import pandas as pd
import numpy as np
import math
import sys
from signal import *
import calendar
from datetime import datetime

def clean(*args): 
	df.to_csv(csv_to_read + '.csv')
	print "\nPROCESSING COMPLETE"
	sys.exit(0);

for sig in (SIGABRT, SIGILL, SIGINT, SIGSEGV, SIGTERM):
    signal(sig, clean)

csv_to_read = 'tung_hist_jan_mar_weather_nolocomotion_mood'
df = pd.read_csv(csv_to_read + '.csv', index_col=0)

for index, row in df.iterrows():
	if math.isnan(row['Mood']):
		print str(index) + ": " + row['Name'] + " " + row["BeginDate"]
		print calendar.day_name[row['WeekDay']] + " " + str(datetime.strptime(row['BeginTime'], "%H:%M:%S").strftime("%I:%M %p")) 
		print row['Duration'][:3]
		print row['Weather']
		while True:
			try:
				mood = input('')
				val = int(mood)
				if val >= 1 and val <= 5:
					break
				else: 
					print "Enter valid input (1-5)"
					continue
			except (TypeError, SyntaxError, ValueError, NameError) as e:
				print "Enter valid input (1-5)"
				continue
		df.at[index, 'Mood'] = mood
		print "\n\n"


