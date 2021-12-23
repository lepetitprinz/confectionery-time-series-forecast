import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dao.OpenAPIWeather import OpenAPIWeather

##############################
# Weather API
##############################
# Initiate Open API for weather data class
api_weather = OpenAPIWeather()

# Get API Information
api_weather.set_date_range()
api_weather.get_api_info()

# # Get API dataset
data_weather = api_weather.get_api_dataset()

# Insert into db of API dataset
print("Save the API dataset")
api_weather.save_result_on_db(data=data_weather)

# data_weather_avg = api_weather.save_avg_weather()
