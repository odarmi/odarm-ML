@echo off

for /l %%x in (21, 1, 21) do (
	start chrome /new-window --app=https://www.google.fr/maps/timeline/kml?authuser=0^&pb=!1m8!1m3!1i2018!2i2!3i%%x!2m3!1i2018!2i2!3i%%x
)
	
