# Capstone Project: Airbnb listings Price prediction and Web application [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://edmundho7-airbnbsg-prediction-app-bwyph3.streamlitapp.com/)

## Executive Summary

### Problem Statement

Airbnb is an online marketplace focused on short-term homestays and experience. Its main business model is a shared economy 

The company acts as a broker and charges a commission from each booking.


With the Covid-19 pandemic over and travellers coming back to Singapore, there is an increased demand for short-term accommodation which Airbnb owners can capitalise and rent out their property for revenue.  
One problem that new Airbnb owners have is on how to optimally price their property to maximise the occupancy and revenue amount. Although Airbnb provide guides on how to price the listings, 
there are currently no free services where users can generate an estimated pricing based on the features of the property. The usual method that homeowners price their property is to find similar listings around the area and 
price it to these properties. Onr problem with this pricing method is that the homeowners might miss out on advantages their property have over others such as amenities like private gym or a Hot tub. 

For this project, I will create a Web application that generate an estimated pricing when users key in their listing information. To create this application, a price prediction model 
using various Supervised machine learning regression algorithms will be developed and the performance of the regression model will be assessed by their R2 Score, RMSE and the generalisation of the model.
The best performing model will be selected as the production model to be used for the web application. Besides providing an estimate for the user, this project also identifies and provide insights on the relationships 
between the various features and the property price. This can assist host in understanding the features and amenities that are crucial for a higher Airbnb price thereby increasing their earnings.
 
## Datasets
The Airbnb listing dataset that I will be using were sourced from [InsideAirbnb.com](http://insideairbnb.com/get-the-data), a mission driven project that provides data and advocacy about Airbnb's impact on residential communities
by scraping and reporting data on Airbnb listings.

The dataset was last scraped on 22 September 2022 and contains information of all the Airbnb listings on the particular day. 

The dataset comprises of the following:
- `Listings`: Detailed listings data of 75 attributes for each of the listings. The data dictionary is shown below for the relevant attributes
- `Calendar`: Detailed calendar data showing listings availability, dates, prices (adjusted, minimum, maximum)
- `Reviews`: Detailed reviews for Listings showing reviewer, comments
- `Neighbourhoods`: GeoJSON file of the different neighbourhoods in Singapore


The data dictionary of the dataset can be found [here](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=1322284596)

##	Exploratory Data Analysis

<img src = './images/waffle_property.png' width='1000'>

<img src = './images/box_neighbourhood.png' width='1000'>

The Airbnb listings in Singapore are made up of 80% apartment, 16% of Hotel and House which contribute a small 4% of the total listings. The apartment listings are scattered throughout the island while the hotels are centered around the CBD area and Southern Waterfront area.

 <img src = './images/map_scatterlistings.png' width='1000'>
 
 The listings in Marina South fetch the highest price among the neighbourhoods at about $600 per night 
while most listings are about $150-$200 per night. 

As for the features, listing prices generally increases with number of guests and beds, and having a sharp drop when accommodation capacity exceed 9 pax. This is due to the fact that those listings are hostels with high capacity and low prices.

<img src = './images/bar_accommodates.png' width='800'>
<img src = './images/bar_beds.png' width='800'>


## Preprocessing & Feature Engineering

Preprocessing & Feature engineering was also performed on the dataset to create additional features to be used for supervised machine learning. 

An example of feature engineering is that the amenities for each listings were grouped into various categories and specific amenities that are likely to influence listing price were selected to determine their effect on listing price. 
In general, consumers expect amenities such as TV, Air conditioning and Security lock to be available, prices tend to be lower when owners do not provide these amenities.

** Insert amenities**

Besides the amenities, word count of the listing and description was also performed as Airbnb owners tend to provide renters with important information such as location, amenities and no. of rooms of their listings in their title and description to increase the clickrates and booking of their listings. 


Similarly to the housing market, proximity to MRT stations and City centre also affects the listing price. Therefore, the distance to the nearest MRT and City centre was calculated via haversine formula.


## Regression model Evaluation/Metrics

|                Model | R2 (Train) | R2 (Test) | RMSE (Train) | RMSE (Test) | Generalisation |
|---------------------:|-----------:|----------:|-------------:|------------:|:--------------:|
| K-Nearest Neighbours |        1.0 |     0.678 |          0.0 |      76.074 | 32.20%         |
|        Random Forest |      0.897 |     0.723 |       52.944 |      74.553 | 19.40%         |
|            Light GBM |      0.958 |     0.767 |       35.306 |      69.126 | 19.94%         |
|              XGBoost |      0.762 |     0.696 |       80.179 |      81.432 | 8.66%          |

The regression models were evaluated based on the metrics. The metrics are as follows:
R2 Score - Higher value indicates how much the model can explain the variance in the listing prices
RMSE - Average deviation between the predicted and actual price
Generalisation score - Higher value indicates the model show signs of overfitting to the data and inability to adapt and react to unseen data 

Based on the above criteria, XGBoost was selected as the best performing model due to its much lower generalisation score compared to the other models. 
Therefore XGBoost will be used for the final production model. 

## Production model Evaluation

As XGBoost is selected to be used for the final production model which have a much lesser number of features that are user inputs. The production model will be compared with the original full featured XGBoost model
to see the performance difference between the two.

|                   Model | R2 (Train) | R2 (Test) | RMSE (Train) | RMSE (Test) | Generalisation |
|------------------------:|-----------:|----------:|-------------:|------------:|:--------------:|
| XGBoost (Full features) |      0.762 |     0.696 |       80.179 |      81.432 | 8.66%          |
|    XGBoost (Production) |      0.690 |     0.615 |       90.442 |      91.767 | 10.87%         |

A comparison of the full featured model and the reduced model is shown in the table below. 

There is a slight drop in the R2 values with an increase in RMSE for the Train and Test data and an increase in generalisation score from 8.67% to 10.87%.
The model is also only able to explain about 61.5% of the variation in the listing prices with an RMSE price of $91.80 on the test data.


** Insert Scatterplot of Full and Production model**

Comparing the full feature and production models, the full feature model perform fairly wells with majority of the predicted scatter points falling close to the diagonal line (representing perfect prediction) 
for listings below $250 as compared to listings below $120 for the production model. The models also tend to underpredict the prices for listings above $250 and $120 respectively and performed poorly for listings above $300 
with the predicted prices being significantly lower than the actual prices.



On average, the full feature model is able to predict the listings price within $14.8 of the actual price (or about 11.4%) for 70% of the listings while the production model is only able to predict listings price within
$16.60 of the actual price (or about 13.5%) for 60% of the listings.


## Feature importance with SHAP

It is not enough to just create a regression model but to also interpret the machine learning model and derive insights on how the various features affect the prediction result of the model. This can be done using SHAP values which was first proposed by Lundberg and Lee as a unified approach to explain the output of any machine learning model.
The benefits of using SHAP values are that 
1) Global interpretability - It can be used to summarize the impact of each features on the prediction.
2) Local interpretability - It can be used to explain the prediction for a single observation as each observation gets its own SHAP values, allowing us to identify the features that contributed to the prediction.
3) SHAP values can be calculated for any tree-based models

To get an overview of which features are important for the model, we can plot a bar plot of the SHAP values for the features in descending order to determine the importance of the features.

<img src = './images/shap_vip.png' width='800'>

Besides using SHAP to get an overview, we can also use it to understand how each features impact the predicted price of each listing by creating a waterfall plot to show how the model derived the final pricing based on the various features.

**E[f(x)]**: is the output value of the model's prediction for the given input. \
 **f(x)**: is the base value for the given input. It is the price that will be predicted if we did not know any features for the current output. \
**Red bars**: Features that push the listing price ***higher*** \
**Blue bars**: Features that pull the listing price ***lower*** \
**Width of bars**: Importance of the feature. The wider it is, the higher impact is has on the price \
**Values**: The logarithmn value of the features, base inputs and output. To get the value ofhow much the feature affect the price of the listing in $, take the exponential of the value

<img src = './images/shap_waterfall1.png' width='800'>

## Price recommender Web App deployment

The XGBoost model was deployed onto a web app using Streamlit. The app allows users to input their Airbnb features into the app to get a predicted listing price. It shows the map of Singapore, with the location of their listing.
In addition, it also displays a Waterfall chart showing to users how the model derived the final predicted pricing from the input features and how they increase or decrease the price.


## Conclusion and Recommendations

Price prediction model and app is useful for homeowners to maximise their property occupancies and revenues at the same time. 
However, more work can be done to improve the production model further to generate a more accurate suggestion

Recommendations:
As the listing price is set by the owner, a more accurate price data such as the actual price paid by the consumers can be used as the target variable
Improve the dataset by incorporating other features such as property size, unit level and proximity features to F&B and attractions
Remove hostels from the data as it has a high number of bedrooms/beds and have lower price than other property types, this affects the model performance

