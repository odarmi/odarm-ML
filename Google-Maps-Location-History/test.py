import process_location as pl
import pyowm


## create dataframe
##pl.create_kml_files('Jan', 1, 'Apr', 10, 'cookie.txt', 'data/')
#df = pl.full_df('data/')
#df.to_csv('tung_hist_jan_mar.csv', sep=',', encoding='utf-8')

#add weather data
owm = pyowm.OWM('8b3086ca196b2447a9f5d373f1747d52')
observation = owm.weather_at_zip_code('75075')
w = observation.get_weather()
print w.get_wind()
print w.get_humidity()

