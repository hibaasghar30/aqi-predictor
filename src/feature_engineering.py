from datetime import datetime
from src.aqi_util import pm25_to_aqi   #use our aqi formula from aqi_util file of pm25 function
from src.feature_store import get_last_row






#weather_data is a raw dict form get_current_weather()
#pollution_data is a raw dict from get_Air_pollution()
#this function turn the two responses (co2 o3 nh3 and the other one humidity temp feelslike etc) into 1 clean row of organized data


def build_feature_row(city_name, weather_data,pollution_data):
  components = pollution_data["list"][0]["components"]  #pollutiondata we get is in layers we have to go through that, and list(co2 o3)  [0] is used so it goes through that list once & saves reslt in variable "components" so we dont need to repeat that again and again
  pm25= components["pm2_5"]   # this ives us just the pm25 number out of all the pollutants list

  aqi = pm25_to_aqi(pm25)   #converts raw pm25 value into an actual aqi value
  now= datetime.now()

  previous_row = get_last_row()

#  if previous_row is not None and "timestamp" in previous_row:
       # previous_time = datetime.fromisoformat(previous_row["timestamp"])
      #  hours_elapsed = (now - previous_time).total_seconds() / 3600
     #   if hours_elapsed > 0:
    #        aqi_change_rate = (aqi - previous_row["aqi"]) / hours_elapsed
   #     else:
  #          aqi_change_rate = 0.0
 # else:
#        aqi_change_rate = 0.0




        #------------------------------REPLACCEECECECECECECECECECE

  if previous_row is not None and "timestamp" in previous_row:
    previous_timestamp = previous_row["timestamp"]

    try:
        if isinstance(previous_timestamp, str):
            previous_time = datetime.fromisoformat(previous_timestamp)
        elif isinstance(previous_timestamp, datetime):
            previous_time = previous_timestamp
        elif hasattr(previous_timestamp, "to_pydatetime"):
            previous_time = previous_timestamp.to_pydatetime()
        else:
            previous_time = None

        if previous_time is not None:
            hours_elapsed = (now - previous_time).total_seconds() / 3600

            if hours_elapsed > 0:
                aqi_change_rate = (
                    (aqi - previous_row["aqi"]) / hours_elapsed
                )
            else:
                aqi_change_rate = 0.0
        else:
            aqi_change_rate = 0.0

    except (TypeError, ValueError):
        aqi_change_rate = 0.0

  else:
    aqi_change_rate = 0.0
        
  #dict to keep all the data in one place
  row = {
    "city": city_name,
    "pm2_5": pm25,
     "pm10": components["pm10"],
        "co": components["co"],
        "no2": components["no2"],
        "so2": components["so2"],
        "o3": components["o3"],
        "temperature": weather_data["main"]["temp"],
        "humidity": weather_data["main"]["humidity"],
        "pressure": weather_data["main"]["pressure"],
        "wind_speed": weather_data["wind"]["speed"],
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),
        "aqi": aqi,
        "aqi_change_rate": round(aqi_change_rate, 2),
        "timestamp": now.isoformat(),
  }

  return row





