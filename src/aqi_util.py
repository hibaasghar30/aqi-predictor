#pm25 is tiny solid particles floating in air
PM25 = [
    (0.0, 12.0, 0, 50),         #(pm25 low ,pm25high, aqi low, aqi high for first range)
    (12.1,  35.4, 51, 100),        
    (35.5, 55.4, 101, 150),    
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500) 
]


#create function
#finds the pmi25 value equal to aqi

def pm25_to_aqi(pm25_value):
    for c_low, c_high, aqi_low, aqi_high in PM25:
        if c_low <= pm25_value <= c_high:
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25_value - c_low) + aqi_low
            return round(aqi, 1)
    return None



#print(pm25_to_aqi(74.3)) # test - confirmed working, gives 160.7



