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