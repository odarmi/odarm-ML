
var unirest = require('unirest');
var csv = require('csv-array');
var GoogleKey = "AIzaSyCO82g-a1lQyZGnKxmjUSHWWMbAYGeUcV0";
//aytung94: var DarkSkyKey = "d3cd0f12052016fa205eea2cb27f9e6e";
var DarkSkyKey = "116c22f9267845c021aca44b673668b6";

function csvParser() {
    return new Promise((resolve, rejcet) => {
        csv.parseCSV("../data/tung_hist_jan_mar_weather.csv", function(data){resolve(data)});
    });
}

function getLocation(name, address) {
    return new Promise((resolve, reject) => {
    	unirest.get("https://maps.googleapis.com/maps/api/geocode/json?address=" 
	            + name + address
		    + "&key=" + GoogleKey
	            , function(response){resolve(response.body.results[0].geometry.location)});

    });
}

function getWeather(time, lat, lng) {
    return new Promise((resolve, reject) => {
    unirest.get("https://api.darksky.net/forecast/"
	            + DarkSkyKey + "/"
	            + lat.toFixed(4) + "," 
	            + lng.toFixed(4) + ","
	            + time//.toFixed(0)
	            , function(response){resolve(response.body.currently.icon)});
    });
}

async function main() {
      let csv_data = await csvParser();
      var weather_data = [];
      var prev_address;
      var prev_name;
//      console.log(JSON.stringify(csv_data));
      for ( i in csv_data){
        let time = csv_data[i].IndexTime;
            time = new Date(time);
            time = time.getTime()/1000;
        let address = csv_data[i].Address;
        let name = '';//csv_data[i].Name;
        let loc;
//        console.log(time, name, address + '\n');
//        if (i < 482) {
        if (address != '') {
            loc = await getLocation(address, name);
            prev_address = address;
            prev_name = name;
        } else {
            loc = await getLocation(prev_address, prev_name); 
        }
        lat = loc.lat;
        lng = loc.lng;
        let weather = await getWeather(time, lat, lng);
        weather_data.push(weather);
//	      console.log(JSON.stringify(weather));
        console.log(i + ', '+ weather);
      }
}

main()
// var location = getLocation().then((results) => {console.log("Location: " + results)});

    // geocoding to get lat lng
//    unirest.get("https://maps.googleapis.com/maps/api/geocode/json?address=" 
//	            + '2428+Bigleaf+Ct,+Plano+Tx'
//		    + "&key=AIzaSyCO82g-a1lQyZGnKxmjUSHWWMbAYGeUcV0"
//	            , function(response){console.log("Location: ", response.body.results[0]););
	
    // Access Weather
//    var DarkSkyKey = "d3cd0f12052016fa205eea2cb27f9e6e";
//    unirest.get("https://api.darksky.net/forecast/"
//	            + DarkSkyKey + "/"
//	            + lat.toFixed(4) + "," 
//	            + lng.toFixed(4) + ","
//	            + time.toFixed(0)
//	            , function(response){console.log(i, "Weather: ", response.body.currently.summary);});
//	console.log("\n");

  //}

//fs.readFile("../data/tung-hist_jan_mar.kml", 'utf8', function (err, data) {
//	var dataArray = data.split(/\r?\n/);
//})



/*
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
*/
