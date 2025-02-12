from import_libraries import * 

#Taking Ticker from user 
def model_training(stock_symbol="IOC.BO", epochs=10, MODAL="LSTM"):

    data_duration='4y'
    test_split_data=0.3

    stock_symbol = stock_symbol  #India Oil Corporation Limited stock ticker symbol
    tkr = yf.Ticker(stock_symbol)  #Create ticker object
    # df =  yf.download(stock, start, end)
    data = tkr.history(period=data_duration)
    data = pd.DataFrame(data)

    # data.head()

    data.reset_index(inplace=True)
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    # data.head()

    data["Date"]=pd.to_datetime(data["Date"]).dt.date
    # data.tail()

    # sns.pairplot(data,x_vars="Date",y_vars="Open")
    # plt.show()

    Ms=MinMaxScaler()

    data.set_index("Date",inplace=True)

    df=data.copy(deep=True)

    df[df.columns]=Ms.fit_transform(df)

    # df.head(3)

    def create_sequence(df):
        sequence = []
        labels = []
        start_idx = 0

        for stop_idx in range(1,len(df)):
            sequence.append(df.iloc[start_idx:stop_idx])

            labels.append(df.iloc[stop_idx])

            start_idx+=1

        return np.array(sequence), np.array(labels)

    #Tain, Test Split 

    number=int(test_split_data*len(df))
    train_data=df[:number]
    test_data=df[number:]

    #Sequencing the data 
    train_seq,train_label = create_sequence(train_data)
    test_seq, test_label = create_sequence(test_data)

    def next_day_prediction(model, scaler, data):
        last_sequence = data.tail(1)
        last_sequence_scaled = scaler.transform(last_sequence)
        last_sequence_reshaped = np.array([last_sequence_scaled])
        tomorrow_predicted = model.predict(last_sequence_reshaped)
        tomorrow_inverse_predicted = scaler.inverse_transform(tomorrow_predicted)
        last_date = data.index[-1]
        tomorrow_date = last_date + pd.Timedelta(days=1)
        tomorrow_prediction_df = pd.DataFrame(tomorrow_inverse_predicted,
                                            columns=["Open_pred", "High_pred", "Low_pred", "Close_pred", "Volume_pred"],
                                            index=[tomorrow_date])
        return tomorrow_prediction_df
    #Definng the model attribute
    model=None
    if MODAL=="GRU":
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))
        model.add(Dropout(0.1))
        model.add(GRU(units=50))
        model.add(Dense(5,activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error','accuracy'])
        model_summary=model.summary()
        model.fit(train_seq, train_label, epochs=int(epochs),validation_data=(test_seq, test_label), verbose=1)
    # test_predicted = model.predict(test_seq)
    else:
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))
        model.add(Dropout(0.1))
        model.add(LSTM(units=50))
        model.add(Dense(5,activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error','accuracy'])
        model_summary=model.summary()
        model.fit(train_seq, train_label, epochs=int(epochs),validation_data=(test_seq, test_label), verbose=1)
    # test_inverse_predicted = Ms.inverse_transform(test_predicted)

    tomorrow_prediction_df=next_day_prediction(model=model,scaler=Ms,data=data)
    # Evaluate the model
    loss, mae, accuracy = model.evaluate(test_seq, test_label)

    # print(f"Test Loss: {loss}")
    # print(f"Test MAE: {mae}")
    # print(f"Test Accuracy: {accuracy}")

    final_accuracy = accuracy

    return final_accuracy, tomorrow_prediction_df, model_summary