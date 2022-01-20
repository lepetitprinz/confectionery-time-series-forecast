import os
import sys
import datetime
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from dao.OpenAPIWeather import OpenAPIWeather

##############################
# Weather API
##############################
# Check start time
print("Start Time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Initiate Open API for weather data class
api_weather = OpenAPIWeather()

# Get API Information
api_weather.get_api_info()

# Get API Information
api_weather.set_date_range()

# # Get API dataset
data_weather = api_weather.get_api_dataset()

# Insert into db of API dataset
print("Save the API dataset")
# print(data_weather)
api_weather.save_result_on_db(data=data_weather)

# Check end time
print("End Time: ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
