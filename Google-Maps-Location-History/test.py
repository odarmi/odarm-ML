import process_location as pl

#pl.create_kml_files('Jan', 1, 'Apr', 10, 'cookie.txt', 'data/')
df = pl.full_df('data/')
df.to_csv('tung_hist_jan_mar.csv', sep=',', encoding='utf-8')
