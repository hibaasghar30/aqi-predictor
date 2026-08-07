import requests   #help python send requests to to websites/APIs through internet
from src import config





#it turns a cityname into longitude/latitude coordinates because weather API needs it not just the city name
def geocode(city_name, country_name):
    url="https://api.openweathermap.org/geo/1.0/direct"
    params= {
        "q": f"{city_name},{country_name}",
        "limit": 1,
        "appid": config.OPENWEATHER_API_KEY,
    }
    response = requests.get(url, params=params)
    data = response.json()
    lat = data[0]["lat"]  #data[0] grabs the first item from the lsit
    lon = data[0]["lon"]
    return lat, lon





#another function to find weather
def get_current_weather(lat,lon):
    url="https://api.openweathermap.org/data/2.5/weather"
    params={
        "lat":lat,
        "lon":lon,
        "appid": config.OPENWEATHER_API_KEY,
        "units":"metric",    #gives us weather in celsius as openweather gives in kelvin as default
    }
    response = requests.get(url, params=params)
    return response.json()

#testing
#lat, lon = geocode("Karachi", "PK")
#print(lat, lon)

#weather = get_current_weather(lat, lon)
#print(weather)





#function to find about the pollution
# fetches pollutant concentrations (PM2.5, PM10, CO, etc) for a given location
def get_air_pollution(lat,lon):
    url=("http://api.openweathermap.org/data/2.5/air_pollution")
    params={
        "lat": lat,
        "lon": lon,
        "appid": config.OPENWEATHER_API_KEY,
    }
    response = requests.get(url, params=params)
    return response.json()

#uses geocode function to print long lati of a city
lat, lon = geocode("Karachi", "PK")
print(lat, lon)
#uses get current weather function gives temp in celsius 
weather = get_current_weather(lat, lon)
print(weather)
#uses get air pollution function
pollution = get_air_pollution(lat, lon)
print(pollution)


def get_air_pollution_history(lat, lon, start, end):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": lat,
        "lon": lon,
        "start": start,
        "end": end,
        "appid": config.OPENWEATHER_API_KEY,
    }
    response = requests.get(url, params=params)
    return response.json()

#from src.aqi_util import pm25_to_aqi
#print(pm25_to_aqi(25.45))