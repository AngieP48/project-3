# AI Bootcamp Group Project 3
## Sentiment Stock Price Analysis

### Project ideation
Through this project we will:
1. Analyze stock price impacts and the potential correlation of company/stock sentiment based on related Tweets
   
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

- Data fetching - Kaggle
- Data exploration
- Data transformation
- Data analysis
  Must haves: Additional columns
  - Number of tweets per day
  - Sentiment score per day
  - Difference between close to close
  - Difference on open to close
  Nice to have
  - Rolling 30, 60, 90 day return
  - different indicators on stock price
- Data cleaning and preprocessing
  - Feature engineering
  - Sentiment indicator (Positive, Neutral, Negative, Total Tweets)
  - Reshape data into multi-index and apply aggregations
  - OneHotEncode
  - OPENAI API prompt engineering for sentiment analysis
- Testing ML models
  - Long Short Team (LST) memory models
    - Used in time series analysis
- Integrate AI tools into the project for deployment
  - Time Series Forecasting; with Prophet
    - Use data correlation to evaluate the predictive relationship among time series patterns
  - Gradio Input story / grade setiment / output future stock price
- Creating documentation
- Creating the presentation
