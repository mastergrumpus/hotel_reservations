# Predicting Hotel-Reservation Cancellations
**Norman Jen & Yamuna Umapathy**

## Business Problem & Understanding
**Stakeholders:** CEO of Flatiron Hotels

The online Hotel reservations have dramatically changed booking possibilities and customer's behaviours. Most of the time hotel booking
cancellations can be hurtful to business owners, although it is often made easier at a low cost and beneficial to hotel guests. Last minute 
cancellations can result in loss of revenue unless some measures are undertaken to mitigate the loss.

Goal: The purpose of this project is to analyze Hotel Bookings data and investigate cancellations. We are predicting the likelihood
of cancellations when booking reservation is made, our goal is to find the best Machine Learning model.

### Dataset and Data Exploration:

For analysis, we are using a Hotel Reservation Dataset from Kaggle: https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset/data

Our data set comes from 2 hotels in Portugal, which provided very complete data on approximately 36,000 previous reservations throughout 2017 and 2018. There was no data missing whatsoever and there were little to no misspellings, erroneous values, or non-sensical data points for us to deal with.

Numeric Columns    : no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights, lead_time, no_of_previous_cancellations, 
                     no_of_previous_bookings_not_canceled, avg_price_per_room.

Categorical Columns: type_of_meal_plan, required_car_parking_space, room_type_reserved, arrival_year, arrival_month, arrival_date,
                     market_segment_type, repeated_guest, booking_status.

### Data Cleaning Before Preprocessing:

After exploring our Dataset, We added some new columns, changed a few feature's categorical values to have numerical values before preprocessing to reduce overfitting, following below was completed as part of the process.

 - Columns with month date and year was concatenated to find `arrival_date`, which was used to find `booking_date`.
 - Created new column `total_guests`, `length_of_stay`
 - Categorical columns such as `type_of_meal_plan`, `room_type_reserved`, `market_segment_type` & `booking_status` was replaced with numerical.
 - Dropped columns `Booking_ID`, `arrival_day`, `booking_date` which may not be used later.

Overall, our dataset was very clean and required no imputation. However, we discovered that there were 37 records with arrival dates of 2/29/2018. Since 2018 was not a leap year, this date does not exist. We did not want to imbalance the 2/28/2018 or 3/1/2018 dates and since we did not have information on what these dates were meant to represent and we had a very healthy amount of data, we felt comfortable eliminating those results.

### Preprocessing:

Train_Test_Split was used to split our Train samples and Test samples before preprocessing to avoid Data Leakage. We applied Standard Scaler to
scale the numeric columns, and One Hot Encoder for categorical columns.

**Target y**:`booking_status` was chosen as our Target variable.

**X**       : Remaining columns except `booking_status`.

**Correlation**

Before applying the data to our models, we wanted to take a preliminary look at the correlation between the features in our data set and our target variable.

We see that some features have a high correlation such as `lead_time` and `no_of_special_requests`. Others are less important but do have a small degree of correlation.

Given this information, we decided to split our data into 2 separate train-test splits - one with all features included, and one with only features with correlations of ~0.1 or higher. We ran all models with both splits to see the effect of having more complete vs. more streamlined feature sets on the model's performance.

<p align="center">
    <img src = "https://github.com/YamunaU75/Hotel-Reservations/blob/main/Data/Heatmap.png" width = "750" height="451">
</p>

### Models Choice and Validation:

We chose 3 different classifier models to use - Logistic Regression, Decision Trees, and Random Forest Classifier. All 3 specialized in classification through different methods, each with distinct advantages and disadvantages, in particular regarding prediction, probability, and over-fitting. We implemented all 3 to make sure we used the model with the high predictive power without over-fitting.

False Positive - In this case, our model would classify a reservation as being likely to be canceled in the future, when the guest never makes a cancellation.
False Negative - In this case, our model would classify a reservation as likely to be maintained, but is ultimately canceled. Although not ideal and hopefully minimized, it is understood that this will happen occasionally. 

1. **Baseline Model Logistic Regression**:
   Using `booking_status` as y variable, remaining columns as X, and with following parameters: fit_intercept = False, max_iter = 1000,
   C = 1e5, solver = 'liblinear', random_state=100. We got **AUC score 0.87** as results.

3. **Logistic Regression with High Correlated Features**:
   Using `booking_status` as y varaiable, and high correlated features as X, and with same parameters as baseline model: fit_intercept = False,    max_iter = 1000, C = 1e5, solver = 'liblinear', random_state=100. We got **AUC score 0.85** as results.

<p align="center">
    <img src = "https://github.com/YamunaU75/Hotel-Reservations/blob/main/Data/ROC_logistic.png" width = "750" height="550">
</p>

3. **Decision Tree with all features & Optimized Hyperparameter**:
   By Hyperparameter tuning to find best parameters, we found Max Tree depth 8, Min Sample splits 0.01, Min Sample Leafs 0.01 & Max Feature 12
   for Decision Tree. Building Tree model with following parameters, we got **AUC score 0.80** which was far away from Logistic Baseline Model.

4. **Decision Tree with feature selection & Optimized Hyperparameter**:
   By choosing tuned hyperparameters and high correlated features: Max Tree depth 8, Min Sample splits 0.01, Min Sample Leafs 0.01 & Max Feature 12.
   Building Tree model with following parameters and feature selected, we got **AUC score 0.83** not better than Logistics results.


6. **Oversampling Technique, Logistic Regression**:
   Target variable has slightly imbalanced classes, Class 0 as 67% and Class 1 as 33%. We want to check if Oversampling technique will
   leads us to best model, we still got **AUC score as 0.85** .

7.  **Random Forest Classifier Baseline**:
    Using `booking_status` as y variable, remaining columns as X, and with parameters:n_estimators=100, criterion = 'entropy', random_state=100.
    We got the best **AUC score 0.93**, but was seeing our Test results are bad when Train results are doing good.

8. **Random Forest Classifier with High Correlated Features**:
    Using `booking_status` as y variable, high correlated features as X, and with parameters:n_estimators=100, criterion = 'entropy',                               random_state=100. We were getting the best **AUC score 0.92**, but again seeing our Test results are bad when Train results are
    doing good.

9. **Random Forest Classifier, Optimal Hyperparameter**:
   Random Tree Classifier was improving our results, but model was overfitting with Baseline and High correlated feature models.
   We want to check tuning hyperparameter and found parameters: Max Tree depth 13, Min Sample splits 0.01, Min Sample Leafs 0.01,
   Max Feature 19 and applying n_estimators = 100. We got better **AUC score 0.8807** and better train and test scores.

10. **Random Forest Classifier, Optimal Hyperparameter & High Correlated Features**:
   Random Tree Classifier was improving our results, but model was overfitting with Baseline and High correlated feature models. Finally,
   we want to check tuning hyperparameter with high correlated features and used parameters: Max Tree depth 13, Min Sample splits 0.01,
   Min Sample Leafs 0.01, Max Feature 19 and applying n_estimators = 100. We got best **AUC score 0.8850** and best train and test scores.

### Visualizing the results of Hyperparameter Tuning

Max Depth - 13 and 17 were both potential values for us to examine. 13 had the lower AUC score, and 17 had a higher AUC score but beyond that, we were seeing higher potential to over-fit without any significant gain in AUC score.

Minimum Sample Leafs and Minimum Sample Splits - Both of these hyperparameters showed a very steep decline in AUC score with any increase in value. We kept both hyperparameters to 0.01 in our model.

Max Features - Again, we wanted to keep our max features value to a reasonable number. 14 and 19 represented 2 local maximums for us to examine.

<p align="center">
    <img src = "https://github.com/YamunaU75/Hotel-Reservations/blob/main/Data/hyperparameter_random1.png" width = "750" height="267">
</p>
<p align="center">
    <img src = "https://github.com/YamunaU75/Hotel-Reservations/blob/main/Data/hyperparameter_random2.png" width = "750" height="267">
</p>


## Conclusion:
      
   Our final model, Random Forest Classifier with feature selection and tuned hyperparameters, produced the best results without over-fitting
   our train data. Our AUC of 0.8850 shows a solid confidence in our classification. We used this metric because we wanted to measure overall 
   performance to measure both false positives and false negatives, with a slight bias towards the positive class to be safe.

 <p align="center">
    <img src = "https://github.com/YamunaU75/Hotel-Reservations/blob/main/Data/ROC%20final.png" width = "750" height="586">
</p>  

## Next Steps:
- Examine models for Repeated Guest
- Gather more data
- Contingency plan for False positives
- Financial Impact Report
- Survey for Guest Cancellations
- Implement model in real time.

    
## Repository Structure

```
├── Data
├── .gitignore
├── Flatiron Hotels.pdf
├── README.md
└── Hotel.ipynb
```
