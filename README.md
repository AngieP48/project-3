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


---

   - *Nice to have*  
      - *Rolling 30, 60, 90 day return*
      - *Various stock price indicators*




OPENAI API prompt engineering for sentiment analysis  
Testing ML models  
Long Short Team (LST) memory models  
   Used in time series analysis  
Integrate AI tools into the project for deployment  
   Time Series Forecasting; with Prophet  
      Use data correlation to evaluate the predictive relationship among time series patterns  
      Gradio Input story / grade setiment / output future stock price  
Creating documentation  
Creating the presentation  


1. Using VADER (e.g., SentimentIntensityAnalyzer):
Do not remove stopwords: VADER is designed to work well with the full sentence, including stopwords. Removing stopwords could disrupt the natural language structure and potentially lead to less accurate sentiment scores. VADER uses a combination of heuristics and lexical features that include consideration of common words, negations, and other contextual elements that stopwords help provide.

About the Scoring
The compound score is computed by summing the valence scores of each word in the lexicon, adjusted according to the rules, and then normalized to be between -1 (most extreme negative) and +1 (most extreme positive). This is the most useful metric if you want a single unidimensional measure of sentiment for a given sentence. Calling it a 'normalized, weighted composite score' is accurate.

It is also useful for researchers who would like to set standardized thresholds for classifying sentences as either positive, neutral, or negative. Typical threshold values (used in the literature cited on this page) are:

positive sentiment: compound score >= 0.05
neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
negative sentiment: compound score <= -0.05
NOTE: The compound score is the one most commonly used for sentiment analysis by most researchers, including the authors.

The pos, neu, and neg scores are ratios for proportions of text that fall in each category (so these should all add up to be 1... or close to it with float operation). These are the most useful metrics if you want to analyze the context & presentation of how sentiment is conveyed or embedded in rhetoric for a given sentence. For example, different writing styles may embed strongly positive or negative sentiment within varying proportions of neutral text -- i.e., some writing styles may reflect a penchant for strongly flavored rhetoric, whereas other styles may use a great deal of neutral text while still conveying a similar overall (compound) sentiment. As another example: researchers analyzing information presentation in journalistic or editorical news might desire to establish whether the proportions of text (associated with a topic or named entity, for example) are balanced with similar amounts of positively and negatively framed text versus being "biased" towards one polarity or the other for the topic/entity.
IMPORTANTLY: these proportions represent the "raw categorization" of each lexical item (e.g., words, emoticons/emojis, or initialisms) into positve, negative, or neutral classes; they do not account for the VADER rule-based enhancements such as word-order sensitivity for sentiment-laden multi-word phrases, degree modifiers, word-shape amplifiers, punctuation amplifiers, negation polarity switches, or contrastive conjunction sensitivity.