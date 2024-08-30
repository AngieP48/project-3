# AI Bootcamp Group Project 3
## Sentiment Stock Price Analysis

### Project ideation
Through this project we will:
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

Preprocessing:  
1. Data fetching  
2. Data exploration  
   - Basic understanding  
   - Check for missing values  
3. Clean the data  
4. Data transformation  
   - Convert data types  
   - Encode categorical variables  
   - Normalize data  
5. Feature engineering  
   - Create new features  
      - Number of tweets per day  
      - Sentiment score per day (Positive, Neutral, Negative)  
      - Stock closing price difference from previous day  
      - Stock open to close price difference  
   - Select relevant features  
   - Split the data  

   *- Nice to have*  
     *- Rolling 30, 60, 90 day return*  
     *- Various stock price indicators*  




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
