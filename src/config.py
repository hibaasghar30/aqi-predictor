#os and pathlib are built-in to python noneed to pip install them
#need to pip install python-env 
import os                              #os helps to talk with the operating system 
from pathlib import Path               #pathlib provides a clean way to work with files
from dotenv import load_dotenv         #gives specific ability to read env files

load_dotenv()                          #read env file and loads its values into the environment


# os.getenv("AQI_CITY", "Karachi") means:
#  look for a value  AQI_CITY in .env / environment variables,
#  if it's not found then use 'Karachi' as the default

CITY_NAME= os.getenv("AQI_CITY", "Karachi")   #city for which u want to fetch AQI  #karachi is set as default
COUNTRY_CODE = os.getenv("AQI_COUNTRY", "PK") #helps API find the right city

#***************************************************************************************'''
#if you want to add more cities make dictionary 
# CITIES ={["name": "Karachi","Country:" "PK"]},{"name":"Lahore","Country":"PK"]}'''
#***************************************************************************************'''

#API KEY
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "") #look for OPENWEATHER_API_KEY in .env file, default to empty string if missing

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME", "")

#file paths (where to put our data)

PROJECT_ROOT= Path(__file__).resolve().parent.parent    #.parent points to one previous folder/finds main project folder
DATA_DIR= PROJECT_ROOT/"data"         #path to the data folder
MODEL_DIR = PROJECT_ROOT/"models"     #path to the model folder
DATA_DIR.mkdir(exist_ok=True)     #creates data folder if it doesnot exit
MODEL_DIR.mkdir(exist_ok=True)       #creates modelfolder if it doesnt exist


FEATURE_STORE_FILE = DATA_DIR/"feature_store.parquet"      #collected weather/aqi is saved her
MODEL_FILE =MODEL_DIR/"best_model.joblib"                   #trained model is saved here
MODEL_METADATA_FILE =MODEL_DIR / "model_metadata.json"     #info about model(accuracy,features)



#AQI CATEGORIES
HAZARD_ALERT_THRESHOLD =150  #AQI=>150 gives warning

#AQI SCALE (COLORS)
#create a list
AQI_COLORS = [
    (0,50 , "Good, Have a fun day outside","#00e400"),
    (51,100, "Moderate, mostly fine to go outside","#ffff00"),
    (101,150, "Unhealthy for sensitive groups, be a little careful", "#ff7e00"),
    (151,200, "Unhealthy, better to not make plans", "#ff0000"),
    (201,250, "Very unhealthy, stay indoors if you can", "#8f3f97"),
    (251, 500, "Hazardous, avoid going outside", "#7e0023")
    ]

#function creation

def get_aqi_category(aqi_value):
    for low, high, label, color in AQI_COLORS:
        if low <= aqi_value <= high:
         return label, color
    return "Hazardous, avoid going outside", "#7e0023" 


#pm25 is tiny solid particles floating in air inn aqi_util file
