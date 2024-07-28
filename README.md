# California_house_price_estimation
## Data
The dataset used in this project is the California Housing Prices dataset. It is available in the data directory as housing.csv. This dataset includes various features that can be used to predict housing prices, which is analogous to predicting customer churn in terms of methodology.

## Preprocessing
The preprocessing steps are crucial for preparing the data for machine learning models. The steps include:

## Handling Missing Values: Missing data can significantly impact model performance. We impute missing values using appropriate strategies like median imputation to ensure no information is lost.

## Encoding Categorical Variables: Machine learning models require numerical input. We convert categorical variables into numerical values using techniques like one-hot encoding.

## Feature Scaling: Different features may have different scales, which can affect model performance. We use standard scaling to ensure all features contribute equally to the model training process.

We define the preprocessing pipeline using Scikit-learn's Pipeline and ColumnTransformer to ensure a systematic and repeatable process.

## Model Training
We trained several machine learning models on the preprocessed data to find the best performer. The models include:

## Linear Regression: A simple and interpretable model that assumes a linear relationship between features and the target variable.

## Decision Tree Regressor: A non-linear model that splits the data into branches to make predictions. It can capture complex patterns but is prone to overfitting.

## Random Forest Regressor: An ensemble model that combines multiple decision trees to improve performance and reduce overfitting.

## Support Vector Regressor (SVR): A model that uses hyperplanes to make predictions. It is effective for high-dimensional data but requires careful tuning.

## Model Evaluation

# We evaluated the models using the following metrics to understand their performance:

## Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values. Lower values indicate better performance.

## Root Mean Squared Error (RMSE): The square root of MSE, providing an error metric in the same units as the target variable.

## Mean Absolute Error (MAE): Measures the average absolute difference between predicted and actual values. It is less sensitive to outliers compared to MSE.

## Fine-Tuning
To optimize the model performance, we fine-tuned the hyperparameters of the Random Forest Regressor using Grid Search with Cross-Validation. This method systematically tests different combinations of hyperparameters to find the best set that minimizes the error.

## Results
The model evaluations and fine-tuning results are summarized below:

## Linear Regression
RMSE: 68628.19819848922
MAE: 49439.89599001898
## Decision Tree Regressor
RMSE: 0.0 (indicating overfitting, as the model perfectly fits the training data but likely performs poorly on unseen data)
## Random Forest Regressor
RMSE: 18603.515021376355
## Support Vector Regressor (SVR)
RMSE: 111094.6308539982
## Cross-Validation Scores
We used cross-validation to assess model performance more robustly. The results are:

## Decision Tree Regressor
Mean RMSE: 71407.68766037929
Standard Deviation: 2439.4345041191004
## Linear Regression
Mean RMSE: 69052.46136345083
Standard Deviation: 2731.674001798347
## Random Forest Regressor
Mean RMSE: 50182.303100336096
Standard Deviation: 2097.0810550985693
