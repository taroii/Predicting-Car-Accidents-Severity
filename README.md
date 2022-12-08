# Predicting Car Accident Severity
UCLA STATS 101C (Introduction to Statistical Models and Data Mining)  
By Taro Iyadomi, Anish Dulla, and Nishant Jain  
11/18/2022 - 12/02/2022  
  
## 1. **Description**

For this project, we are tasked with predicting the severity of car accidents (mild or severe) in the United States using a country-wide car accident dataset.   

## 2. **Dataset**

The accident data are collected from February 2016 to Dec 2021, using multiple APIs that provide streaming traffic incident (or event) data. These APIs broadcast traffic data captured by a variety of entities, such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors within the road-networks. Currently, there are about 2.8 million accident records in this dataset.  

The dataset consists of a mix of 43 numerical and categorical predictors and a categorical response variable called "Severity". We are free to import other datasets to improve our predictions.  

Approximately 90% of the accidents are labeled "mild". Therefore, in order to build a successful model, our model must achieve a test accuracy of over 90%.  

## 3. **Exploratory Data Analysis and Data Cleaning**

#### Understanding Missing Values

The training dataset contained 13,211 missing values. However, only 17 out of the 43 original predictors accounted for all of them. On top of that, when we look more closely at the structure of those missing values, we find that most of them come from weather related variables, which raises questions about the data collection process, as weather variables weren't being strictly recorded.  

<img src="https://user-images.githubusercontent.com/114524578/206526610-6340fbf8-b417-4d1f-bc6a-48eddee0ae2f.png" width=500 height=300 />

Thankfully, the majority of these predictors had less than 5% missing values, which makes imputation appropriate, as we can increase the amount of information fed to the model without drastically increasing bias.  

##### Top 8 Predictors with Most NAs  

| Variable | % Missing |
|---|---|
| Wind Chill  | 16.19  |
| Wind Speed  |  5.34 |
| Humidity  |  2.42  |
| Wind Direction | 2.41 |
| Visibility | 2.34 |
| Weather Condition | 2.31 |
| Temperature | 2.29 | 
| Pressure | 1.91 |

Before we get to imputation, however, we can first remove some missing values by cleaning the data and reducing multicollinearity between the predictors.  

#### Description

The Description variable shows the natural language descriptions of each accident. These are written reports, and do not follow a specific format. To better understand this data, we created word maps to understand which words are used most frequently overall and which words are used most frequently for severe car accidents. 

All Accidents | Severe Accidents
:------------:|:----------------:
<img src="https://user-images.githubusercontent.com/114524578/205473036-3b9f3a15-03d3-4a02-a7a6-0d0a3e5d0ddd.png" width=400 height=300 /> | <img src="https://user-images.githubusercontent.com/114524578/205473047-98bdd346-3f01-4663-816d-797e66869e3b.png" width=400 height=300 />

Here, we see that while words like "accident" and "road" are common words in both sets, words like "closed" and "blocked" appear more often in the severe car accidents than in all of the car accidents. So, we created dummy variables for these words in place of the description predictor. 

#### Start_Time, End_Time, Weather_Timestamp

The original time variables had far too many levels.  

To fix this, we extracted the hour, month, and year values from the time character strings, then converted those strings into POSIX date time format, which is the number of seconds since January 1, 1970.  

The original time variables were heavily correlated, so we replaced the start/end time variables with a time lapsed variable. This new predictor measures the duration of each accident in seconds.  

Finally, we created a Night variable, which is a binary predictor measuring whether or not the accident took place between 7pm and 7am. This is useful because not only do the variables Sunset_Sunrise, Nautical_Twilight, Astronomical_Twilight, and Civil_Twilight measure the same thing using different nighttime metrics, they also contain missing values, often together in the same observations.  

Since the new Night variable doesn’t contain any missing values, we can combine all of these predictors into one that takes the majority of the 5 predictors. This leaves us with a new Night predictor that contains zero missing values.  

#### Start/End Latitude and Longitude



