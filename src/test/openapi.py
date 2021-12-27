from dao.OpenAPIWeather import OpenAPIWeather
from dao.OpenAPIDust import OpenAPIDust

##############################
# Weather API
##############################
# Initiate Open API for weather data class
api_weather = OpenAPIWeather()

# Get API Information
api_weather.get_api_info()

api_weather.set_date_range()

# # Get API dataset
data_weather = api_weather.get_api_dataset()
# Insert into db of API dataset

print("Save the API dataset")
api_weather.save_result_on_db(data=data_weather)

#
# data_weather_avg = api_weather.save_avg_weather()

##############################
# Dust API
##############################
# # Initiate Open API for dust data class
# api_dust = OpenAPIDust()
#
# # Get API Information
# api_dust.get_api_info()
#
# # Get API dataset
# data_dust = api_dust.get_api_dataset()
#
# # Insert into db of API dataset
# print("Save the API dataset")
# api_dust.save_result(data=data_dust)
#
