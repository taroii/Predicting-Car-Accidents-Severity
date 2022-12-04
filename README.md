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

The data consists of 20 categorical and 24 numerical variables. Many of the categorical variables (or factors) have an enormous amount of categories (levels) that aren't useful for modeling. It's clear that these variables need to be cleaned up.  

#### Description

The Description variable shows the natural language descriptions of each accident. These are written reports, and do not follow a specific format. To better understand this data, we created word maps to understand which words are used most frequently overall and which words are used most frequently for severe car accidents. 

All Accidents | Severe Accidents
:------------:|:----------------:
<img src="https://user-images.githubusercontent.com/114524578/205473036-3b9f3a15-03d3-4a02-a7a6-0d0a3e5d0ddd.png" width=400 height=300 /> | <img src="https://user-images.githubusercontent.com/114524578/205473047-98bdd346-3f01-4663-816d-797e66869e3b.png" width=400 height=300 />


