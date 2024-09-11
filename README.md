# AI Bootcamp Group Project 3
## Sentiment Stock Price Analysis

### Project ideation
Through this project we will predict stock prices based on sentiment created by company/stock related tweets  

1. Analyze stock price impacts and the potential correlation of company/stock sentiment based on related same day Tweets  
   | Ticker | Company |
   | :-: | - |  
   | AAPL | Apple Inc. |  
   | MSFT | Microsoft Corporation |  
   | PG | Procter & Gamble Company |  
   | TSLA | Tesla, Inc. |  
   | TSM | Taiwan Semiconductor Manufacturing Company Limited |  
3. Analyze and predict future stock closing price based on sentiment (Positive, Neutral, Negative)

Data: [Stock Tweets for Sentiment Analysis and Prediction](https://www.kaggle.com/datasets/equinxx/stock-tweets-for-sentiment-analysis-and-prediction?resource=download)  
License: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)

### Preprocessing:
   - Data fetching
   - Data exploration
      - Basic understanding
      - Check for missing values
   - Clean the data
      - Handle missing values
      - Handle outliers
   - Data transformation
      - Convert data types
      - Encode categorical variables
      - Normalize data
   - Feature engineering
      - Create new features
         - Number of tweets per day
         - Sentiment score per day (Positive, Neutral, Negative)
         - Stock closing price difference from previous day
         - Stock open to close price difference
      - Select relevant features
      - Split the data


**Prophet Modeling**
- Import dependencies
- Read in CSV's
- Stock Data Preprocessing
  - Understanding the df shape and columns
  - Assessing df data types
  - Filtered to specific stocks for project
  - Checked for null and blank values
- Tweet Data Preprocessing
  - Understanding the df shape and columns
  - Assessing df data types
  - Filtered to specific stocks for project
  - Checked for null and blank values
- Implementation of VADER Sentiment Analyzer
  - Initialized Sentiment Analyzer
  - Created a function to analyze each tweet
   - Returned a sentiment label based on the scoring scale:
      - positive sentiment: compound score >= 0.05
      - neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
      - negative sentiment: compound score <= -0.05
   - Added Positive, Neutral, Compound and Sentiment columns
   - Grouped tweet data by Date and Stock Name to arrive at a Mean Compound score for each date, renaming the Compound column to Mean Compound
- Implemented initial training and testing loop for each stock
   - Mapped 'Sentiment' values
   - Split training and testing data
   - Added 'Mean Compound' regressor
   - Fit model
   - Create future dataframe
   - Predict future values
   - Calculate and display metrics for each loop
- Implemented secondary training and testing loop for each stock
   - Mapped 'Sentiment' values
   - Created a 60-day rolling average as an additional regressor
   - Split training and testing data
   - Added regressors:
      - 'Mean Compound
      - 'Volume'
      - 'Sentiment'
      - '60-day rolling average'
   - Fit model
   - Create future dataframe
   - Predict future values
   - Calculate and display metrics for each loop
- Implmented Prophet Model with Visualizations
   - Mapped 'Sentiment' values
   - Created a 60-day rolling average as an additional regressor
   - Added regressors:
      - 'Mean Compound
      - 'Volume'
      - 'Sentiment'
      - '60-day rolling average'
   - Fit Predict
   - Merge the df with regressors
   - Fill null or zero values
   - Forecast
   - Create visualizations

**CNN Modeling**
- Import libraries
- Data Preprocessing:
  - Define the stock tweet dataframe from csv file
  - Define the stock_yfinance dataframe from csv file
  - Create and apply a date function to derive the data from a datetime column
  - Create and apply function using Vader to generate a sentiment score based on the tweets in the stock_tweet dataframe
  - Feature Engineering:
    - Create daily count column to tally the number tweets a company has daily
    - Create open/close diff column to calculate the difference between open and close
    - Create Prev Close Diff column to calculate the difference between prev Close price
  - Merge stock_tweet and stock_yfinance dataframe
  - Isolate Close column since this is the target
  - Separate stock dataframe into numerical and categorical dataframes for normalization.
  - Normalize numerical data using StandardScaler
  - Normalize categorical data using pandas get_dummies function with the dtype set to int
  - Merge categorical and numerical dataframes to get final_stock_df
  - Fill NA values
  - Save final_stock_df to csv final_stock_data.csv
- Model Creation:
  - Import libraries
  - Define stock_df dataframe from final_stock_data.csv
  - Define X with all features from the stock_df except the Close feature
  - Define y as the Close feature
  - Split X and y values into train and test data
  - Define the CNN model using linear regression as the focus.
    - Created 7 Dense layers in total where the input layer has 18 nodes and the output layer has one node.
    - Activation for model was set to 'relu' for all layer except the output layer which was set to 'linear'
  - Compile the model with the loss metric set to mean absolute percentage error as accuracy is not a good metric for this problem.
  - Fit and train the model, which had a loss of roughly 0.15 percent and a mean absolute error of 0.39
  - The R2 score which is used to compare how close the predicted y values are to the true y values is: 
  0.9999610781669617
- Model Tuning: 
  - Defined a create_model function that uses the same methodology as my previous model, but adjust the hyper parameters per run.
  - Created and ran a keras_tuner variable that runs the previously defined function 60 times to get the best model and hyperparameters.
  - With those results, we calculated the best model's R2 score which is: 0.9990188479423523
  - Then I compared the y_test and y_pred values and they are mostly within 1 points difference.
- Model Application: 
  - Define a predict function to take in user inputed data to process and predict the Close value.
  - Test the code
  - Define the application using gradio and run.


**LSTM MODEL**
- Import Libraries
- Data Preprocessing
   - Define the stock tweet data frame from the CSV file (stock_tweets.csv).
   - Define the stock_yfinance dataframe from the CSV file (stock_yfinance_data.csv).
    - Create and apply a date function to extract the date from a datetime column in both dataframes.
   - Apply VADER Sentiment Analysis to generate a sentiment score based on the tweets in the stock_tweet dataframe.
- Feature Engineering
   - Create a daily count column to tally the number of tweets a company receives daily.
   - Create an open/close diff column to calculate the difference between the opening and closing stock prices.
   - Create a previous close diff column to calculate the difference between the previous day's closing price and the current day's closing price.
- Merging and Cleaning:
    - Merge the stock_tweet and stock_yfinance dataframes on Date and Stock Name.
    - Isolate the Close column since this is the target variable for prediction.
    - Separate the stock dataframe into numerical and categorical dataframes for normalization.
    - Normalize the numerical data using MinMaxScaler.   
    - Merge the categorical and numerical dataframes to create merged_data.
    - Drop NA
    - Save merged_data to CSV (merged_sentiment_stock_data.csv) for use in model training.
- Function Structure
    - analyze_tweets(): Analyzes the sentiment of user-inputted tweets.
   - create_sequences(): Creates sequences of 60 days of stock data for LSTM training.
   - fetch_recent_stock_data(): Fetches stock data using yfinance.
    - train_model(): Trains the LSTM model on the selected stock data.
   - predict_next_day_close_yfinance(): Predicts the next day's closing stock price using the trained LSTM model.
   - predict_stock_price(): Gradio output function for final price
   - Gradio Interface: Combines two interactive tabs:
       - Stock Price Prediction Tab: For training the model and predicting stock prices.
       - Tweet Sentiment Analysis Tab: For analyzing the sentiment of multiple tweets.
- Model Creation   
   - Define stock_df dataframe by loading the preprocessed final_stock_data.csv.
   - Define X and y variables:
    X: All features from stock_df except the Close column.
    y: The Close column (target variable).
   - Split X and y values into training and testing datasets using train_test_split.
- LSTM Model Structure
     - Define the LSTM model:
    - Used a sequential model with:
        - LSTM layer with 50 units to capture time-series dependencies.
        - Dropout layer with a 20% dropout rate to prevent overfitting.
        - Dense layer with 25 units for dense representation.
        - Output layer with 1 unit to predict the stock’s closing price.
         - Activation function: Used relu for all layers except the output, which uses linear.
- Compile the model:
    - Set the loss function to mean squared error (MSE) and used the Adam optimizer.
- Train the model:
    - Trained on the dataset with X_train and y_train, using a batch size of 1 and running 10 epochs. The validation set was X_test and y_test. 
- Model Application
     - This function takes user-input data (such as stock symbol, sentiment score, tweet count, open/close price) and processes it to predict the next day's closing stock price.





