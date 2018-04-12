
//var googleMapsClient = require('@google/maps').createClient({
//  key: 'AIzaSyCO82g-a1lQyZGnKxmjUSHWWMbAYGeUcV0',
//  Promise
//});
//
//var response = googleMapsClient.geocode({
//  address: '1600 Amphitheatre Parkway, Mountain View, CA',
//}).asPromise();
//console.log(response);

// Geocode an address.
//googleMapsClient.geocode({
//  address: '1600 Amphitheatre Parkway, Mountain View, CA'
//}, function(err, response) {
//  if (!err) {
//    console.log(response.json.results);
//  }
//});


var fs = require("fs");
//console.log("\n *STARTING* \n");
var contents = fs.readFileSync("../data/tung_hist_03_21_18.json");
var jsonContent = JSON.parse(contents);

var i = 0;
for (i = 80; i < 90; i++) {
	var lat = jsonContent.locations[i].latitudeE7   / 10000000.0;
	var lng =  jsonContent.locations[i].longitudeE7 / 10000000.0;
	var time = jsonContent.locations[i].timestampMs / 1000;
	var date = new Date(0);
	date.setUTCSeconds(time);
	
	console.log("First Place Latitude: ", lat);
	console.log("First Place Longitude: ", lng);
	//console.log("First Place Time: ", time);
	console.log(i, " Date: ", date);
	//console.log("\n *EXIT \n");
	
	// Unirest variable
	var unirest = require('unirest');
	
	// Access Google Unique Address
	unirest.get("https://maps.googleapis.com/maps/api/geocode/json?latlng=" 
	            + lat + "," 
	            + lng  
		    + "&key=AIzaSyCO82g-a1lQyZGnKxmjUSHWWMbAYGeUcV0"
	            , function(response){console.log(i, "Location: ", response.body.results[0].place_id);});
	
	// Access Weather
	var DarkSkyKey = "d3cd0f12052016fa205eea2cb27f9e6e";
	unirest.get("https://api.darksky.net/forecast/"
	            + DarkSkyKey + "/"
	            + lat.toFixed(4) + "," 
	            + lng.toFixed(4) + ","
	            + time.toFixed(0)
	            , function(response){console.log(i, "Weather: ", response.body.currently.summary);});
	console.log("\n");
}	

