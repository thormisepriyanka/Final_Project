import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
from keras.models import load_model
import pickle

def create_sequence(df):
    sequence = []
    labels = []
    start_idx = 0

    for stop_idx in range(1,len(df)):
        sequence.append(df.iloc[start_idx:stop_idx])

        labels.append(df.iloc[stop_idx])

        start_idx+=1

    return np.array(sequence), np.array(labels)

pred_columns=["Open_pred", "High_pred", "Low_pred", "Close_pred", "Volume_pred"]
data_columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# pred_columns=["Open_pred", "High_pred", "Low_pred", "Close_pred"]
# data_columns=['Date', 'Open', 'High', 'Low', 'Close']

def test_seq_and_label(df,model,scaler):
    df3=df.tail(int(1300*0.30)) #total market working day from Jan 2020 to Dec 2025
    df3=df3[data_columns]
    Ms=scaler
    df3.set_index("Date",inplace=True)
    df3[df3.columns]=Ms.transform(df3)
    test_data=df3
    test_seq , test_label = create_sequence(test_data)
    loss, mae, accuracy = model.evaluate(test_seq, test_label)
    # print(f"Test Loss: {loss}")
    # print(f"Test MAE: {mae}")
    # print(f"Test Accuracy: {accuracy}")
    final_accuracy = accuracy
    return final_accuracy



# App Title
st.title("📈 Stock Chronos: Time Series Forecasting")

# User Input: Stock Symbol
stock_symbol = st.text_input("TickerName / Stock Name (e.g : AAPL)")

# Load Stock Data
if stock_symbol:
    tkr = yf.Ticker(stock_symbol)
    df = tkr.history(period="1y")
    df.reset_index(inplace=True)
    df["Date"]=pd.to_datetime(df["Date"]).dt.date
    # Display Stock Data
    st.write("### 📊 Stock Data Preview")
    st.dataframe(df.tail())

    # Plot Closing Price
    st.write("### 📈 Stock Closing Price Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Close'], label="Close Price", color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()
    st.pyplot(fig)

    #Stock Price Prediction 
    if stock_symbol=="IOC.BO":
        # df1 = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df1 = df[data_columns]
        df1.set_index('Date',inplace=True)

        model = load_model("LSTM_Next_day_prediction.h5")
        with open("LSTM_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        def next_day_prediction(model, scaler, data):
            last_sequence = data.tail(1)
            last_sequence_scaled = scaler.transform(last_sequence)
            last_sequence_reshaped = np.array([last_sequence_scaled])
            tomorrow_predicted = model.predict(last_sequence_reshaped)
            tomorrow_inverse_predicted = scaler.inverse_transform(tomorrow_predicted)
            last_date = data.index[-1]
            tomorrow_date = last_date + pd.Timedelta(days=1)
            tomorrow_prediction_df = pd.DataFrame(tomorrow_inverse_predicted,
                                                columns=pred_columns,
                                                index=[tomorrow_date])
            return tomorrow_prediction_df

        tommorows_predition = next_day_prediction(model,scaler,df1)
        tommorows_predition1 = tommorows_predition.copy(deep=True)
        tommorows_predition.reset_index(drop=True,inplace=True)
        tommorows_predition=pd.DataFrame(tommorows_predition)
        st.write("#### 📊 Tommorows Forecasting Using LSTM")
        st.dataframe(tommorows_predition1)
        final_accuracy =  test_seq_and_label(df=df,model=model,scaler=scaler)
        st.write(f"Prediction Accuracy: {int(final_accuracy*100)}%")
    else:
        st.write (f"#### 📊 Tommorows Forecasting Using LSTM: No Data available for the {stock_symbol}")

    if stock_symbol=="IOC.BO":
        df2 = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df2.set_index('Date',inplace=True)

        model_GRU = load_model("GRU_Next_day_prediction.h5")
        with open("GRU_scaler.pkl", "rb") as f:
            scaler_GRU = pickle.load(f)

        def next_day_prediction(model, scaler, data):
            last_sequence = data.tail(1)
            last_sequence_scaled = scaler.transform(last_sequence)
            last_sequence_reshaped = np.array([last_sequence_scaled])
            tomorrow_predicted = model.predict(last_sequence_reshaped)
            tomorrow_inverse_predicted = scaler.inverse_transform(tomorrow_predicted)
            last_date = data.index[-1]
            tomorrow_date = last_date + pd.Timedelta(days=1)
            tomorrow_prediction_df = pd.DataFrame(tomorrow_inverse_predicted,
                                                columns=pred_columns,
                                                index=[tomorrow_date])
            return tomorrow_prediction_df

        tommorows_predition = next_day_prediction(model,scaler,df2)
        tommorows_predition1 = tommorows_predition.copy(deep=True)
        tommorows_predition.reset_index(drop=True,inplace=True)
        tommorows_predition=pd.DataFrame(tommorows_predition)
        st.write("#### 📊 Tommorows Forecasting Using GRU")
        st.dataframe(tommorows_predition1)
        final_accuracy =  test_seq_and_label(df=df,model=model_GRU,scaler=scaler_GRU)
        st.write(f"Prediction Accuracy: {int(final_accuracy*100)}% ")
    else:
        st.write (f"#### 📊 Tommorows Forecasting Using GRU: No Data available for the {stock_symbol}")



## remove the comma"," from the volume before training 
#round all number the value to 2

# Round all values to 2 decimal places
# df = df.round(2)
