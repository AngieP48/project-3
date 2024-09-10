import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from sklearn.model_selection import train_test_split
import random

def get_score(tweet):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(tweet)['compound']
    return score

def get_date(date):
    return date[0:10]

def add_features(df):
    stock_df = df

    stock_df['Datetime'] = stock_df['Date'].apply(get_date)

    date_stock_df = stock_df.groupby(['Datetime','Stock Name']).size().reset_index(name='Tweet Count')

    stock_df = pd.merge(stock_df,date_stock_df,
                           on=['Datetime','Stock Name'],
                           how='inner'
                           )
    
    stock_df['Open/Close Diff'] = round(abs(stock_df['Open'] - stock_df['Close']),2)

    stock_df['Prev Close Diff'] = random.uniform(100,400)

    return stock_df

def numerical_pre_processing(df):
    close_df = df['Close']
    numerical_df = df.select_dtypes(['int64','float64'])
    numerical_df = numerical_df.drop(columns=['Close'])
    ct = ColumnTransformer([('scaler',StandardScaler(),numerical_df.columns)])
    array = ct.fit_transform(numerical_df)
    scaler = MinMaxScaler(feature_range=(0,1))
    array = scaler.fit_transform(array)
    numerical_df = pd.DataFrame(data=array,columns=ct.get_feature_names_out())

    return numerical_df, close_df


def categorical_pre_processing(df):
    stock_list = ['TSLA', 'MSFT', 'PG', 'META', 'AMZN']
    categorical_df = df.select_dtypes('object')
    for a in stock_list:
        categorical_df['Stock Name_{a}'] = 0
    stock = categorical_df['Stock Name'][0]
    categorical_df['Stock Name_{stock}'] = 1
    
    return categorical_df




    
    

    


    
