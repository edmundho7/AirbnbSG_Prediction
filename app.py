import streamlit as st
import shap
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import geopy as gp
from geopy.geocoders import Nominatim

from PIL import Image
import haversine as hs
from haversine import Unit

from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBRegressor 

st.set_page_config(layout="wide")

# -------------------------------------------------------------------------------------------------------------------------------------------------

st.sidebar.title("Input your Airbnb property information")


def user_input_features():
# Let us get the user input
    from geopy.geocoders import Nominatim
    address_input = st.sidebar.text_input("Enter the Address of your property", "79 Anson Road")
    geolocator = Nominatim(user_agent="airbnb")
    loc = geolocator.geocode(address_input + ", Singapore")
    # Convert the user address into Latitude, Longitude
    user_lat = loc.latitude
    user_long = loc.longitude

    # The rest of the user input
    property_input = st.sidebar.selectbox('Select the property type', ('Apartment', 'Hotel', 'House'))
    room_input = st.sidebar.selectbox('Select the room type', ('Entire home/apt', 'Private room', 'Shared room','Hotel room'))
    bedrooms_input = st.sidebar.number_input('Select the number of bedrooms', 1, 5, 1)
    beds_input = st.sidebar.number_input('Select the number of beds', 1,50, 1)
    bathroom_input = st.sidebar.number_input('Select the number of bathrooms', 1, 20, 1)
    bathtype_input = st.sidebar.selectbox('Select the bathroom type', ('private', 'shared','baths'))

    data = {'latitude': user_lat,
            'longitude': user_long,
            'property_type': property_input,
            'room_type': room_input,
            'bedrooms': bedrooms_input,
            'beds': beds_input,
            'bathroom_qty': bathroom_input,
            'bathroom_type': bathtype_input
            }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Create prediction button
btn = st.sidebar.button("Get estimated listing price")

# Let us upload the Airbnb icon image
image = Image.open('./images/Airbnb-logo.png')
# Center the image
col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image(image, use_column_width = True)
with col3:
    st.write(' ')

html_temp = """
<div style="background:#ff5c5c ;padding:10px">
<h2 style="color:white;text-align:center;"> Price Prediction App </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)
st.subheader('Created by Edmund Ho (DSIF 5)')
#st.subheader('Problem Statement')
st.markdown('''
At Airbnb, we have noticed that homeowners are often unsure of the price to set for their property. 

This is especially true for new hosts who are unsure of the market price for their property. 

This app aims to help homeowners to set a price for their property based on the features of their property and also provide insights on the factors that affect the price of their property.
''')

with st.expander("How to use this application"):
    st.markdown("""Input the features of your Airbnb on the *left sidebar* to have the model predict the price of your listing (shown below the map). 
    If a particular feature is not known, just estimate or leave it as default. Feel free to play around with the values to see the range of prices.
""")
st.write('---')

# Load the original Airbnb listings data

def load_data():
    listings_modelling = pd.read_csv('./data/listings_user_inputs.csv')
    return listings_modelling

listings_modelling = load_data()

# Drop the target variable from the original dataset
X = listings_modelling.drop('price', axis=1)


# Concat the user input with the original Airbnb listings data
df_final = pd.concat([input_df, X], axis = 0)

# Load the mrt data
@st.cache
def load_mrt():
    mrt = pd.read_csv('./data/mrt_data.csv')
    return mrt

mrt = load_mrt()

# Find the distance from each listing to the nearest MRT station using the Haversine formula
mrtCoord = mrt[['lat', 'lng']]
listingsCoord = df_final[['latitude', 'longitude']]
mrtDisp = [min([hs.haversine((listing[0],listing[1]), (station[0],station[1]), unit=Unit.KILOMETERS) for station in mrtCoord.values.tolist()]) for listing in listingsCoord.values.tolist()]
mrtIdx = [np.argmin([hs.haversine((listing[0],listing[1]), (station[0],station[1]), unit=Unit.KILOMETERS) for station in mrtCoord.values.tolist()]) for listing in listingsCoord.values.tolist()]
mrtName = [mrt['station_name'].iloc[idx] for idx in mrtIdx]
df_final["mrtDisp"] = mrtDisp
df_final["nearestMRT"] = mrtName

# Find the distance from each listing to the City centre using the Haversine formula
# Create a dataframe for City centre coordinate 
city_centre = pd.DataFrame({'city_centre': ['City Centre'], 'lat': [1.2833], 'lng': [103.8500]})
# Find the distance from each listing to city centre
cityCoord = city_centre[["lat", "lng"]]
listingCoord = df_final[["latitude", "longitude"]]
cityDisp = [min([hs.haversine((listing[0],listing[1]), (city[0],city[1]), unit=Unit.KILOMETERS) for city in cityCoord.values.tolist()]) for listing in listingCoord.values.tolist()]
df_final["cityDisp"] = cityDisp

# Dummy encoding for categorical variables
encode = ['property_type', 'room_type', 'bathroom_type','nearestMRT']
# One hot encoding
df_final = pd.get_dummies(df_final, columns=encode)
# Get the user input data
user_input = df_final[:1]

# Scale numeric variables
# numeric = ['latitude','longitude', 'bedrooms', 'beds', 'bathroom_qty', 'mrtDisp', 'cityDisp']
# scaler = StandardScaler()
# scaler.fit(user_input[numeric])

  
# Load in model
model = pickle.load(open('./models/final_xgb.pkl', 'rb'))

#Display listing location on map
st.subheader('Location of your listing')
userCoord = user_input[["latitude", "longitude"]]
st.map(userCoord)
# Adjust map size

st.write('---')

# Apply model to make predictions
prediction = model.predict(user_input)
# As the model is trained on log(price), we need to convert the prediction back to price
prediction = np.exp(prediction)
prediction = '$ {:,.2f}'.format(float(prediction))

# Display the predicted price
st.header('Predicted Listing Price')
st.subheader(prediction)

st.write('---')

# Explain the model's prediction using SHAP values

#(Hide Streamlit pyplot deprecation warning)
st.set_option('deprecation.showPyplotGlobalUse', False)

explainer = shap.Explainer(model)
shap_values = explainer(user_input)

st.subheader('Explanation of the model prediction using SHAP values')
st.write('Which features have the biggest impact on the price of a listing?')

with st.expander("See explanation for understanding SHAP values"):
    st.write("""
             """)
    st.markdown("""

**E[f(x)]**: is the output value of the model's prediction for the given input.

 **f(x)**: is the base value for the given input. It is the price that will be predicted if we did not know any features for the current output.
**Red bars**: Features that push the listing price ***higher***

**Blue bars**: Features that pull the listing price ***lower***

**Width of bars**: Importance of the feature. The wider it is, the higher impact is has on the price

**Values**: The logarithmn value of the feature. To get how much the feature affect the price of the listing in $, take the exponential of the value
""")

# Plot a waterfall chart
plot1 = shap.plots.waterfall(shap_values[0])
plt.gcf().set_size_inches(6, 3)
# Set font size
plt.rcParams.update({'font.size': 8})
st.pyplot(plot1, bbox_inches='tight')
