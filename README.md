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
The Airbnb listing dataset that I will be using were sourced from InsideAirbnb.com (http://insideairbnb.com/get-the-data), a mission driven project that provides data and advocacy about Airbnb's impact on residential communities
by scraping and reporting data on Airbnb listings.

The dataset was last scraped on 22 September 2022 and contains information of all the Airbnb listings on the particular day. 

The dataset comprises of the following:
- `Listings`: Detailed listings data of 75 attributes for each of the listings. The data dictionary is shown below for the relevant attributes
- `Calendar`: Detailed calendar data showing listings availability, dates, prices (adjusted, minimum, maximum)
- `Reviews`: Detailed reviews for Listings showing reviewer, comments
- `Neighbourhoods`: GeoJSON file of the different neighbourhoods in Singapore

| Field                                        | Type     | Description                                                                                                                                                                             |
|----------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| id                                           | integer  | Airbnb's unique identifier for the listing                                                                                                                                              |
| name                                         | text     | Name of the listing                                                                                                                                                                     |
| description                                  | text     | Detailed description of the listing                                                                                                                                                     |
| host_id                                      | integer  | Airbnb's unique identifier for the host/user                                                                                                                                            |
| host_name                                    | text     | Name of the host. Usually just the first name(s).                                                                                                                                       |
| host_response_time                           | text     | average   amount of time the host takes to reply to guest inquiries and booking   requests                                                                                              |
| host_response_rate                           | text     | proportion   of guest inquiries and booking requests that the host replies to                                                                                                           |
| host_acceptance_rate                         | text     | That rate at which a host accepts booking requests.                                                                                                                                     |
| host_is_superhost                            | boolean  | Whether host is superhost                                                                                                                                                               |
| host_has_profile_pic                         | boolean  | Whether host has profile picture                                                                                                                                                        |
| host_identity_verified                       | boolean  | Whether host identity is verified                                                                                                                                                       |
| neighbourhood                                | text     | The neighbourhood of the listing as specified by host                                                                                                                                   |
| neighbourhood_cleansed                       | text     | The neighbourhood as geocoded using the latitude and longitude against   neighborhoods as defined by open or public digital shapefiles.                                                 |
| neighbourhood_group_cleansed                 | text     | The neighbourhood group as geocoded using the latitude and longitude   against neighborhoods as defined by open or public digital shapefiles.                                           |
| latitude                                     | numeric  | Uses the World Geodetic System (WGS84) projection for latitude and   longitude.                                                                                                         |
| longitude                                    | numeric  | Uses the World Geodetic System (WGS84) projection for latitude and   longitude.                                                                                                         |
| property_type                                | text     | Self selected property type. Hotels and Bed and Breakfasts are described   as such by their hosts in this field                                                                         |
| room_type                                    | text     | [Entire home/apt\|Private   room\|Shared room\|Hotel]                                                                                                                                   |
| accommodates                                 | integer  | The maximum capacity of the listing                                                                                                                                                     |
| bathrooms                                    | numeric  | The number of bathrooms in the listing                                                                                                                                                  |
| bathrooms_text                               | string   | The number of bathrooms in the   listing.       On the Airbnb web-site, the bathrooms field has evolved from a number to a   textual description. For older scrapes, bathrooms is used. |
| bedrooms                                     | integer  | The number of bedrooms                                                                                                                                                                  |
| beds                                         | integer  | The number of bed(s)                                                                                                                                                                    |
| amenities                                    | json     | Amenities available in the listing                                                                                                                                                      |
| price                                        | currency | Daily price in local currency                                                                                                                                                           |
| availability_30                              | integer  | Number of nights available to be booked in the next 30 days                                                                                                                             |
| availability_60                              | integer  | Number of nights available to be booked in the next 60 days                                                                                                                             |
| availability_90                              | integer  | Number of nights available to be booked in the next 90 days                                                                                                                             |
| availability_365                             | integer  | Number of nights available to be booked in the next 365 days                                                                                                                            |
| number_of_reviews                            | integer  | The number of reviews the listing has                                                                                                                                                   |
| review_scores_rating                         | float    | Average overall review rating scores                                                                                                                                                    |
| review_scores_accuracy                       | float    | Average rating scores for listing description accuracy                                                                                                                                  |
| review_scores_cleanliness                    | float    | Average rating scores for property cleanliness                                                                                                                                          |
| review_scores_checkin                        | float    | Average rating scores for guests' check-in process                                                                                                                                      |
| review_scores_communication                  | float    | Average rating scores for host's communication                                                                                                                                          |
| review_scores_location                       | float    | Average rating scores for listing location                                                                                                                                              |
| review_scores_value                          | float    | Average rating scores for value-for-money consideration                                                                                                                                 |
| instant_bookable                             | boolean  | Whether or not the property can be instant booked, without having to   message the host first and wait to be accepted                                                                   |
| calculated_host_listings_count               | integer  | The number of listings the host has in the current scrape, in the   city/region geography.                                                                                              |
| calculated_host_listings_count_entire_homes  | integer  | The number of Entire home/apt listings the host has in the current   scrape, in the city/region geography                                                                               |
| calculated_host_listings_count_private_rooms | integer  | The number of Private room listings the host has in the current scrape,   in the city/region geography                                                                                  |
| calculated_host_listings_count_shared_rooms  | integer  | The number of Shared room listings the host has in the current scrape, in   the city/region geography                                                                                   |
| reviews_per_month                            | numeric  | The number of reviews the listing has over the lifetime of the listing                                                                                                                  |





##	Data Imbalance

As the given data is heavily imbalanced, with 95% of the data indicated as negative WNV, resulting in poor performance from predicting the positive WNV cases.  
In order to address the issue of data imbalance, an over-sampling method called SMOTE (Synthetic Minority Oversampling Technique) is adopted for all the classifier models. As such, AUC score will be the main metrics used to assess the best model as compared accuracy.

## Model Evaluation/Metrics
|   Model |   Train score |   Test score |   Generalisation |   Accuracy |   Precision |   Recall |   Specificity |      F1 |   ROC AUC |
|--------------:|--------------:|-------------:|-----------------:|-----------:|------------:|---------:|--------------:|--------:|----------:|
|Logistic Regression (no SMOTE|         0.947 |        0.946 |            0.106 |      0.946 |       0.5     |    0.015     |         0.999 | 0.029     |    0.8203 |
|Logistic Regression|         0.957 |        0.928 |            3.030  |      0.928 |       0.253 |    0.182 |         0.970  |   0.212 |    0.7949 |
|Random Forest Classification|         0.951 |        0.866 |           8.938 |      0.866 |       0.171 |    0.394 |         0.892 |   0.238 |    0.8208 |
|Decision Tree Classification|         0.985 |        0.878 |           10.863 |      0.878 |       0.142 |    0.255 |         0.913 |   0.182  |    0.5997 |
|Multinomial Naive Bayes Classification|         0.834 |        0.785 |            5.875 |      0.785 |       0.152 |    0.664          |0.791 |   0.247 |    0.8181 |
|k-Nearest Neighbour Classification|         0.901 |        0.638 |           29.190 |      0.638 |       0.088 |    0.613 |         0.640 |   0.154 |    0.6813  |
|AdaBoost Classification|        0.960 |        0.913 |            4.167 |      0.913 |       0.224 |    0.241 |         0.950 |   0.242 |    0.8328 |
|XGBoost Classification|         0.969 |        0.889 |            8.256 |      0.889 |       0.187 |    0.321 |         0.921 |   0.236 |    0.8319 |
|Gradient Boosting Classification|         0.981 |        0.915 |            6.728 |      0.915 |       0.199 |    0.197 |         0.955 |   0.198 |    0.8000 |

Classification models, unlike regression models have multiple metrics which results may be based on. Accuracy is used when the True Positives and True negatives are more important while F1-score is used when the False Negatives and False Positives are crucial. F1 score is usually handy for imbalanced datasets (as its calculated based on Precision and Recall). Lastly, for the area under curve of a ROC curve, it measures the usefulness of a test in general, where a greater area means a more useful test, the areas under ROC curves are used to compare the usefulness of tests. AUC metric measure performance for the classification problem at various threshold settings.

For the purpose of solving our problem statement, we will be considering the following criterias to select the best performing model:

1. ROC AUC score > 0.8 
2. Generalisation < 5% 
3. F1 score 

<br>Based on the above criterias, we have selected AdaBoost+SMOTE. 

- **ROC AUC** of 0.8328 infers that the model has a good capability of classifying the positive class in the dataset.
- **Generalisation Score** of 4.167% means that the model has the ability to properly adapt to new, previously unseen data, drawn from the same distribution as the one used to create the model.
- **Recall** of 0.241. The high recall score means our modell succeeds well in finding all the positive cases in the data, even though it may also wrongly identify some negative cases as positive cases.
- **F1 score** of 0.242. As F1 Score is an average of Precision and Recall, it means that the F1 score gives equal weight to Precision and Recall. All the models does not have a great F1 Score. However, our model has a good balance of both Precision and Recall. 

## Conclusion 
Using ADABoost (our best performing model), we achieved an ROC_AUC score of  0.8328 and F1 score of 0.242. Feature importance of our model showed that location features (Latitude & Longitude) as well as weather features (DewPoint, ResultDir, Temperature & SunHours) ranked the highest. This indicates that WNV is most likely to occur at given locations and under certain weather conditions. Our interpretation for these features to score high could be attributed to denser locations which gives the mosquitoes more opportunities for breeding as well as seasonal cycles where temperatures are ideal for the Culex species to thrive such as Summer. Therefore spray efforts should be concentrated at these locations when weather conditions are right.


## Recommendations
1. Through our cost-benefit analysis, the projected cost of spraying would be financially justified as long as it prevents more than 64 individuals from being hospitalized due to the West Nile Virus.

2. Though our costing assumptions are completely straightforward, we believe that there are other more cost-effective techniques that may be applied in conjunction to spraying such as creating awareness amongst the community. These initiatives may be performed through campaigns, education programs and home visits/checks.

3. Explore further in detail on deploying targeted spray areas from our model predictions. This will in turn help directly reduce the cost of spraying efforts across Chicago (such as the random spray cluster at High Ridge Knolls Park). However, as the current spray datasets does not substantially quantify the spraying efforts, more evidence (from a better designed and documented spraying regime) would be recommended.


