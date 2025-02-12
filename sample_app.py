from main_lstm import *
# from main_gru import *
# App Title

if __name__ == "__main__":
    st.title("📈 Stock Chronos: Time Series Forecasting")


    # # URL for NSE-listed companies (example link)
    # nse_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    # # Read the CSV file
    # tickers_df = pd.read_csv(nse_url)
    # Extract the tickers
    # tickers = tickers_df['SYMBOL'].tolist()   #This will contain the list of stocks ticker in the for of list
    tickers=["IOC.BO","GOOG"]
    choosed_ticker_option = st.selectbox(
        "TickerName / Stock Name (e.g : AAPL)",
        tuple(tickers),
    )
    stock_selected=choosed_ticker_option
    st.write("You selected:",stock_selected)
    # User Input: Stock Symbol
    # stock_symbol = st.text_input("TickerName / Stock Name (e.g : AAPL)")

    tkr = yf.Ticker(stock_selected)
    df = tkr.history(period="1mo")
    df.reset_index(inplace=True)
    df["Date"]=pd.to_datetime(df["Date"]).dt.date
    # Display Stock Data
    st.write("### 📊 Stock Data Preview")
    st.dataframe(df.tail())

    # Plot Closing Price
    st.write("### 📈 Stock Closing Price Over Time")
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(df['Date'], df['Close'], label="Close Price", color="blue")
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Closing Price")
    # ax.legend()
    # st.pyplot(fig)
    # df['Date'] = df['Date'].dt.strftime('%m-%Y')
    st.line_chart(df,x="Date",y=["Open","High","Low","Close"],y_label="Price")


    #Predict the price using model 
    st.write("#### 📊 Tommorows Forecasting Using LSTM")
    st.write(f"Would you like to predict the tommorows mkt price of {stock_selected}?")


    selected_epoch=st.slider("Optimal the epoch higher the accuracy:", 10, 150, 10)
    st.write("Hold tight we are gonna predict the tommorows ticker price")

    predict_using_lstm , predict_using_gru = st.columns(2)
    if predict_using_lstm.button("LSTM",type='primary'):
        st.write("Predicted Using LSTM")
        final_accuracy, tomorrow_prediction_df, model_summary = model_training(stock_symbol=stock_selected,epochs=selected_epoch,MODAL="LSTM") 
        st.dataframe(tomorrow_prediction_df)
        st.write(f"Prediction Accuracy Using LSTM: {int(final_accuracy*100)}%")
        st.write(print(model_summary))
    if predict_using_gru.button("LSTM-GRU",type='primary'):
        st.write("Predicted Using GRU")
        final_accuracy, tomorrow_prediction_df, model_summary = model_training(stock_symbol=stock_selected,epochs=selected_epoch,MODAL="GRU") 
        st.dataframe(tomorrow_prediction_df)
        st.write(f"Prediction Accuracy Using GRU: {int(final_accuracy*100)}%")
        st.write(print(model_summary))

