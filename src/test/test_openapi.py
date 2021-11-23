from dao.OpenAPIWeather import OpenAPIWeather
from dao.OpenAPIDust import OpenAPIDust

###############
# Weather API
###############
# Initiate Open API for weather data class
# api_wea = OpenAPIWeather()

# Get API Information
# api_wea.get_api_info()

# # Get API dataset
# data_wea = api_wea.get_api_dataset()
# # Insert into db of API dataset
print("Save the API dataset")
# api_wea.save_result(data=data_wea)

###############
# Dust API
###############
# Initiate Open API for dust data class
api_dust = OpenAPIDust()

# Get API Information
api_dust.get_api_info()

# Get API dataset
data_dust = api_dust.get_api_dataset()

# Insert into db of API dataset
print("Save the API dataset")
api_dust.save_result(data=data_dust)

